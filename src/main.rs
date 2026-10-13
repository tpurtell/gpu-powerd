//! `gpu-powerd`: run the daemon, or inspect and adjust it.
use anyhow::{Context, Result, anyhow, bail};
use gpu_powerd::backend::{Backend, DeviceInfo};
use gpu_powerd::config::{Config, DEFAULT_CONFIG, LOCK_FILE, Settings};
use gpu_powerd::control::{Controller, quantise};
use gpu_powerd::daemon::{Daemon, fit_settings, load_models, restore_all, write_atomic};
use gpu_powerd::model::ThermalModel;
use gpu_powerd::nvml::Nvml;
use std::fs::File;
use std::io::Write as _;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const USAGE: &str = "\
gpu-powerd: thermal- and budget-aware NVIDIA power-limit daemon

Usage:
  gpu-powerd [--config PATH]               run the daemon (root; systemd unit gpu-powerd.service)
  gpu-powerd --check [--config PATH]       read-only: readings, board ranges and the plan (no root)
  gpu-powerd --restore [--config PATH]     set every GPU to its fallback_w (root; ExecStopPost)
  gpu-powerd status [--json]               the running daemon's state
  gpu-powerd model [--json]                the learned thermal model per GPU
  gpu-powerd set KEY=VALUE... [--config PATH] [--no-reload]
                                           edit the config and reload the daemon (root)
                                           KEY is a top-level key or gpu.SELECTOR.KEY;
                                           KEY= removes the setting
  gpu-powerd --version

Config: /etc/gpu-powerd/config.toml (see config.example.toml). Reload: systemctl reload gpu-powerd.
";

struct Args {
    command: String,
    config: PathBuf,
    json: bool,
    reload: bool,
    rest: Vec<String>,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        command: "run".into(),
        config: PathBuf::from(DEFAULT_CONFIG),
        json: false,
        reload: true,
        rest: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    let mut command_set = false;
    let mut set = |args: &mut Args, command: &str| -> Result<()> {
        if command_set {
            bail!("more than one command given\n\n{USAGE}");
        }
        command_set = true;
        args.command = command.into();
        Ok(())
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" | "help" => set(&mut args, "help")?,
            "-V" | "--version" => set(&mut args, "version")?,
            "--check" | "check" => set(&mut args, "check")?,
            "--restore" | "restore" => set(&mut args, "restore")?,
            "status" | "model" | "set" | "run" => set(&mut args, &arg)?,
            "--json" => args.json = true,
            "--no-reload" => args.reload = false,
            "--config" | "-c" => args.config = it.next().context("--config needs a path")?.into(),
            _ if arg.starts_with("--config=") => args.config = arg["--config=".len()..].into(),
            _ if args.command == "set" && arg.contains('=') => args.rest.push(arg),
            _ => bail!("unexpected argument {arg:?}\n\n{USAGE}"),
        }
    }
    Ok(args)
}

fn main() {
    let result = parse_args().and_then(|args| match args.command.as_str() {
        "help" => {
            print!("{USAGE}");
            Ok(())
        }
        "version" => {
            println!("gpu-powerd {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "check" => check(&args),
        "restore" => restore(&args),
        "status" => status(&args),
        "model" => model(&args),
        "set" => set(&args),
        _ => run(&args),
    });
    if let Err(e) = result {
        eprintln!("gpu-powerd: {e:#}");
        std::process::exit(1);
    }
}

fn is_root() -> bool {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() == 0 }
}

/// The config file, or the defaults when it does not exist (said on stderr).
fn load_config(path: &Path) -> Result<Config> {
    if path.exists() {
        Config::load(path)
    } else {
        eprintln!("gpu-powerd: {} not found; using the defaults", path.display());
        Ok(Config::default())
    }
}

/// An exclusive flock on the lock file. `wait` blocks up to that long. The daemon
/// records its pid there (for `set`); `--restore` does not.
fn lock(wait: Option<Duration>, record_pid: bool) -> Result<File> {
    if let Some(dir) = Path::new(LOCK_FILE).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(LOCK_FILE)
        .with_context(|| format!("open {LOCK_FILE}"))?;
    let deadline = Instant::now() + wait.unwrap_or_default();
    loop {
        // SAFETY: a valid open file descriptor; flock has no memory effects.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            if record_pid {
                file.set_len(0).ok();
                (&file).write_all(format!("{}\n", std::process::id()).as_bytes()).ok();
            }
            return Ok(file);
        }
        if Instant::now() >= deadline {
            let holder = std::fs::read_to_string(LOCK_FILE).unwrap_or_default();
            bail!("another gpu-powerd holds {LOCK_FILE} (pid {})", holder.trim());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// `sd_notify(3)` without libsystemd.
fn notify(message: &str) {
    let Some(path) = std::env::var_os("NOTIFY_SOCKET") else { return };
    let Ok(socket) = std::os::unix::net::UnixDatagram::unbound() else { return };
    let bytes = path.as_encoded_bytes();
    let result = if let Some(name) = bytes.strip_prefix(b"@") {
        use std::os::linux::net::SocketAddrExt;
        std::os::unix::net::SocketAddr::from_abstract_name(name)
            .and_then(|a| socket.send_to_addr(message.as_bytes(), &a))
    } else {
        socket.send_to(message.as_bytes(), &path)
    };
    if let Err(e) = result {
        eprintln!("gpu-powerd: sd_notify failed: {e}");
    }
}

fn open_nvml(wait: Duration) -> Result<Nvml> {
    let deadline = Instant::now() + wait;
    loop {
        match Nvml::open() {
            Ok(nvml) if !nvml.devices().is_empty() => return Ok(nvml),
            Ok(_) if Instant::now() >= deadline => bail!("NVML reports no GPUs"),
            Err(e) if Instant::now() >= deadline => return Err(e),
            _ => std::thread::sleep(Duration::from_secs(1)),
        }
    }
}

enum Event {
    Stop,
    Reload,
}

fn run(args: &Args) -> Result<()> {
    if !is_root() {
        bail!("setting power limits needs root; `gpu-powerd --check` is the read-only view");
    }
    let _lock = lock(None, true)?;
    let (tx, rx) = mpsc::channel();
    let mut signals = signal_hook::iterator::Signals::new([
        signal_hook::consts::SIGTERM,
        signal_hook::consts::SIGINT,
        signal_hook::consts::SIGHUP,
    ])?;
    std::thread::spawn(move || {
        for signal in signals.forever() {
            let event = if signal == signal_hook::consts::SIGHUP { Event::Reload } else { Event::Stop };
            if tx.send(event).is_err() {
                break;
            }
        }
    });
    // The driver may still be coming up at boot.
    let nvml = open_nvml(Duration::from_secs(60))?;
    let config = load_config(&args.config)?;
    let settings = config.resolve(nvml.devices())?;
    describe(nvml.devices(), &settings);
    let stream_socket = settings.config.stream.then(|| settings.config.stream_socket.clone());
    let mut daemon = Daemon::new(nvml, settings);
    // The sample stream for gpu-powerd-ws: a local Unix socket, never a network socket.
    // It is optional: failing to open it is logged, not fatal.
    if let Some(path) = stream_socket {
        match gpu_powerd::stream::Publisher::bind(&path) {
            Ok(mut publisher) => {
                eprintln!("gpu-powerd: publishing samples on {}", path.display());
                daemon.publish = Some(Box::new(move |line: &[u8]| publisher.publish(line)));
            }
            Err(e) => eprintln!("gpu-powerd: cannot open the stream socket {}: {e}", path.display()),
        }
    }
    let period = Duration::from_secs_f64(daemon.settings.config.period_s);
    let mut last = Instant::now();
    let mut ready = false;
    let mut next = Instant::now();
    loop {
        let now = Instant::now();
        let dt = if ready { now.duration_since(last).as_secs_f64().max(1e-3) } else { period.as_secs_f64() };
        last = now;
        // A failed write returns here; dropping the daemon restores every GPU.
        let state = daemon.tick(dt)?;
        let status = state.reason.clone();
        if !ready {
            notify("READY=1");
            ready = true;
        }
        notify(&format!("WATCHDOG=1\nSTATUS={}", status.replace('\n', " ")));
        next += Duration::from_secs_f64(daemon.settings.config.period_s);
        if next < Instant::now() {
            next = Instant::now();
        }
        loop {
            match rx.recv_timeout(next.saturating_duration_since(Instant::now())) {
                Ok(Event::Stop) => {
                    notify("STOPPING=1");
                    eprintln!("gpu-powerd: stopping");
                    let failures = daemon.restore();
                    if !failures.is_empty() {
                        bail!("restore failed: {}", failures.join("; "));
                    }
                    return Ok(());
                }
                Ok(Event::Reload) => {
                    match Config::load(&args.config).and_then(|c| c.resolve(daemon.backend.devices())) {
                        Ok(settings) => {
                            eprintln!("gpu-powerd: config reloaded from {}", args.config.display());
                            describe(daemon.backend.devices(), &settings);
                            daemon.reload(settings);
                        }
                        Err(e) => {
                            eprintln!("gpu-powerd: reload rejected, keeping the running config: {e:#}");
                            daemon.config_error = Some(format!("{e:#}"));
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => bail!("signal thread ended"),
            }
        }
    }
}

fn describe(devices: &[DeviceInfo], settings: &Settings) {
    let c = &settings.config;
    eprintln!(
        "gpu-powerd: budget {:.0} W, period {} s, learning {}, stream {}",
        settings.budget_w,
        c.period_s,
        if c.learn { "on" } else { "off" },
        if c.stream { c.stream_socket.display().to_string() } else { "off".into() }
    );
    for (d, g) in devices.iter().zip(&settings.gpus) {
        eprintln!(
            "gpu-powerd: GPU {} {} ({}) base {:.0} W, max {:.0} W, fallback {:.0} W, sensor-fail {:.0} W \
             (board {:.0}..{:.0} W); throttle_start {:.0} C, target {:.0} C, hard {:.0} C; model {}",
            d.index,
            d.uuid,
            d.pci,
            g.base_w,
            g.max_w,
            g.fallback_w,
            g.sensor_fail_w,
            d.board_min_w,
            d.board_max_w,
            g.throttle_start_c,
            g.target_c,
            g.hard_c,
            match (g.use_model, g.model_may_raise) {
                (false, _) => "off",
                (true, true) => "leads when confident",
                (true, false) => "lowers only",
            }
        );
    }
}

fn restore(args: &Args) -> Result<()> {
    if !is_root() {
        bail!("setting power limits needs root");
    }
    // Wait for a dying daemon to let go, but restore regardless.
    let _lock =
        lock(Some(Duration::from_secs(10)), false).map_err(|e| eprintln!("gpu-powerd: {e:#}; restoring anyway")).ok();
    let mut nvml = open_nvml(Duration::from_secs(5))?;
    // A broken config must not block the failsafe: fall back to the defaults (base = board minimum).
    let settings = load_config(&args.config).and_then(|c| c.resolve(nvml.devices())).or_else(|e| {
        eprintln!("gpu-powerd: {e:#}; restoring with the default config");
        Config::default().resolve(nvml.devices())
    })?;
    let failures = restore_all(&mut nvml, &settings);
    if !failures.is_empty() {
        bail!("restore failed: {}", failures.join("; "));
    }
    Ok(())
}

fn check(args: &Args) -> Result<()> {
    let mut nvml = Nvml::open()?;
    if nvml.devices().is_empty() {
        bail!("NVML reports no NVIDIA GPUs");
    }
    let config = load_config(&args.config)?;
    let settings = config.resolve(nvml.devices());
    let n = nvml.devices().len();
    // The first read includes the fan RPC (every 10 s while running); the second is a
    // typical control period.
    let started = Instant::now();
    let _ = (0..n).map(|i| nvml.read(i)).count();
    let first_ms = started.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let started = Instant::now();
    let readings: Vec<_> = (0..n).map(|i| nvml.read(i).map_err(|e| e.to_string())).collect();
    let read_ms = started.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let devices = nvml.devices().to_vec();
    let settings = match settings {
        Ok(s) => s,
        Err(e) => {
            print_devices(&devices, &readings);
            bail!("config does not fit these GPUs: {e:#}");
        }
    };
    let c = &settings.config;
    let fit = fit_settings(c);
    let models = load_models(&c.model_file, &devices, c.model_window_s);
    let summaries: Vec<_> = models.as_ref().map(|m| m.iter().map(|m| m.summary(&fit)).collect()).unwrap_or_default();
    let usable: Vec<Option<&ThermalModel>> = (0..n)
        .map(|i| models.as_ref().filter(|_| settings.gpus[i].use_model && summaries[i].confident).map(|m| &m[i]))
        .collect();
    // An unbounded step: no rate limits or smoothing history, so the plan is where the
    // limits settle for these readings rather than the first step from the current ones.
    let plans = Controller::new(n).step(&settings, &readings, &usable, 1e9);
    if args.json {
        let gpus: Vec<_> = (0..n)
            .map(|i| {
                serde_json::json!({
                    "device": devices[i],
                    "reading": readings[i].as_ref().ok(),
                    "error": readings[i].as_ref().err(),
                    "limits": settings.gpus[i],
                    "plan": plans[i],
                    "target_limit_w": quantise(plans[i].target_w, c.step_w, settings.gpus[i].base_w, settings.gpus[i].max_w),
                    "model": summaries.get(i),
                })
            })
            .collect();
        let out = serde_json::json!({
            "budget_w": settings.budget_w,
            "read_ms_per_gpu": read_ms,
            "read_with_fans_ms_per_gpu": first_ms,
            "gpus": gpus,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }
    print_devices(&devices, &readings);
    println!(
        "NVML cost per GPU: {read_ms:.3} ms per sample, {first_ms:.2} ms with the slow reads \
         (fan every {} s, T.Limit every {} s)",
        c.fan_read_s, c.tlimit_read_s
    );
    println!();
    println!(
        "Config {}: budget {:.0} W, period {} s.",
        if args.config.exists() { args.config.display().to_string() } else { "(defaults)".into() },
        settings.budget_w,
        c.period_s
    );
    println!("Plan (read-only; nothing is written):");
    for i in 0..n {
        let g = &settings.gpus[i];
        let p = &plans[i];
        let target = quantise(p.target_w, c.step_w, g.base_w, g.max_w);
        println!(
            "  GPU {}: {:.0}/{:.0}/{:.0} C, base {:.0} W, max {:.0} W, fallback {:.0} W -> {} W ({}) \
             [demand {:.0} W, pi {:.0} W{}]",
            devices[i].index,
            g.throttle_start_c,
            g.target_c,
            g.hard_c,
            g.base_w,
            g.max_w,
            g.fallback_w,
            target,
            p.reason,
            p.demand_w,
            p.pi_w,
            p.model_w.map(|m| format!(", model {m:.0} W")).unwrap_or_default()
        );
    }
    match &models {
        Some(_) => {
            println!("Model ({}):", c.model_file.display());
            for (i, s) in summaries.iter().enumerate() {
                println!("  GPU {i}: {}", model_line(s));
            }
        }
        None => println!("Model: none saved yet at {} (the PI controller governs)", c.model_file.display()),
    }
    Ok(())
}

fn print_devices(devices: &[DeviceInfo], readings: &[Result<gpu_powerd::backend::Reading, String>]) {
    for (d, r) in devices.iter().zip(readings) {
        println!("GPU {} {} {} [{}]", d.index, d.name, d.uuid, d.pci);
        println!(
            "  board range {:.0}..{:.0} W{}",
            d.board_min_w,
            d.board_max_w,
            d.default_w.map(|w| format!(", default {w:.0} W")).unwrap_or_default()
        );
        match r {
            Ok(r) => println!(
                "  {:.0} C{}, {:.1} W of {:.0} W limit, {:.0}% util ({:.0}% mem){}{}{}",
                r.temperature_c,
                r.tlimit_c.map(|m| format!(" (T.Limit margin {m:.0} C)")).unwrap_or_default(),
                r.power_w,
                r.limit_w,
                r.utilization_pct,
                r.mem_utilization_pct,
                match (r.sm_clock_mhz, r.mem_clock_mhz) {
                    (Some(sm), Some(mem)) => format!(", SM {sm:.0} / mem {mem:.0} MHz"),
                    _ => String::new(),
                },
                r.fan_pct.map(|f| format!(", fan {f:.0}%")).unwrap_or_default(),
                match r.throttle_mask.map(gpu_powerd::backend::throttle_reasons) {
                    Some(reasons) if !reasons.is_empty() => format!(", clocks: {}", reasons.join(" ")),
                    _ => String::new(),
                }
            ),
            Err(e) => println!("  reading failed: {e}"),
        }
    }
}

fn model_line(s: &gpu_powerd::model::ModelSummary) -> String {
    let cross: Vec<String> = s.r_c_per_w.iter().map(|r| format!("{r:.4}")).collect();
    format!(
        "R self {:.4} C/W, R row [{}] C/W, tau {:.0} s, T_amb {:.1} C, residual {:.2} C, \
         confidence {:.0}%{}, {} samples, power std {:.0} W{}",
        s.r_self_c_per_w,
        cross.join(", "),
        s.tau_s,
        s.t_amb_c,
        s.residual_c,
        s.confidence * 100.0,
        if s.confident {
            " (acting)"
        } else if !s.plausible {
            " (implausible)"
        } else {
            ""
        },
        s.samples,
        s.power_std_w,
        s.fan_mean_pct.map(|f| format!(", mean fan {f:.0}%")).unwrap_or_default()
    )
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn paths(args: &Args) -> (PathBuf, PathBuf) {
    let config = Config::load(&args.config).unwrap_or_default();
    (config.state_file, config.model_file)
}

fn status(args: &Args) -> Result<()> {
    let (state_file, _) = paths(args);
    let state = read_json(&state_file).context("is gpu-powerd running?")?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&state)?);
        return Ok(());
    }
    let f = |v: &serde_json::Value| v.as_f64().map_or("?".into(), |x| format!("{x:.0}"));
    let age = gpu_powerd::daemon::now_unix() - state["updated"].as_f64().unwrap_or(0.0);
    println!(
        "gpu-powerd on {} (pid {}): {}, updated {:.0} s ago",
        state["host"].as_str().unwrap_or("?"),
        state["pid"],
        if state["running"].as_bool() == Some(true) { "running" } else { "stopped" },
        age
    );
    println!("budget {} W, period {} s", f(&state["budget_w"]), state["period_s"]);
    if let Some(e) = state["config_error"].as_str() {
        println!("last reload rejected: {e}");
    }
    for g in state["gpus"].as_array().into_iter().flatten() {
        let r = &g["reading"];
        println!(
            "GPU {} [{}]: limit {} W (base {}, max {}), {} C, {} W, {}% util; {}",
            g["index"],
            g["pci"].as_str().unwrap_or("?"),
            f(&g["limit_w"]),
            f(&g["base_w"]),
            f(&g["max_w"]),
            f(&r["temperature_c"]),
            f(&r["power_w"]),
            f(&r["utilization_pct"]),
            g["plan"]["reason"].as_str().unwrap_or("?")
        );
    }
    if age > 10.0 && state["running"].as_bool() == Some(true) {
        println!("warning: the state is stale; the daemon may have died");
    }
    Ok(())
}

fn model(args: &Args) -> Result<()> {
    let (state_file, model_file) = paths(args);
    let config = Config::load(&args.config).unwrap_or_default();
    // The running daemon's state is the freshest; the saved model is the fallback.
    if let Ok(state) = read_json(&state_file)
        && state["running"].as_bool() == Some(true)
    {
        let models: Vec<_> = state["gpus"].as_array().into_iter().flatten().map(|g| g["model"].clone()).collect();
        if args.json {
            println!("{}", serde_json::to_string_pretty(&models)?);
        } else {
            println!("Thermal model (live, from {}):", state_file.display());
            for (i, m) in models.iter().enumerate() {
                match serde_json::from_value::<gpu_powerd::model::ModelSummary>(m.clone()) {
                    Ok(s) => println!("  GPU {i}: {}", model_line(&s)),
                    Err(_) => println!("  GPU {i}: {m}"),
                }
            }
        }
        return Ok(());
    }
    let text = std::fs::read_to_string(&model_file).with_context(|| format!("read {}", model_file.display()))?;
    let file: gpu_powerd::daemon::ModelFile = serde_json::from_str(&text)?;
    let fit = fit_settings(&config);
    let summaries: Vec<_> = file.models.iter().map(|m| m.summary(&fit)).collect();
    if args.json {
        println!("{}", serde_json::to_string_pretty(&summaries)?);
    } else {
        println!("Thermal model (saved, {}):", model_file.display());
        for (i, (uuid, s)) in file.uuids.iter().zip(&summaries).enumerate() {
            println!("  GPU {i} {uuid}: {}", model_line(s));
        }
    }
    Ok(())
}

/// Apply `KEY=VALUE` edits to a TOML document; the result must still be a valid config.
pub fn edit_config(text: &str, edits: &[String]) -> Result<String> {
    let mut doc: toml_edit::DocumentMut = text.parse().context("parse the config")?;
    for edit in edits {
        let (key, value) = edit.split_once('=').with_context(|| format!("{edit:?}: expected KEY=VALUE"))?;
        // gpu.SELECTOR.KEY: the selector may itself contain dots (a PCI bus id "E1:00.0").
        let key = key.trim();
        let (table_path, leaf) = match key.strip_prefix("gpu.").and_then(|rest| rest.rsplit_once('.')) {
            Some((selector, leaf)) if !selector.is_empty() => (Some(selector), leaf),
            _ if !key.contains('.') => (None, key),
            _ => bail!("{key:?}: expected a top-level key or gpu.SELECTOR.KEY"),
        };
        let table = match table_path {
            None => doc.as_table_mut(),
            Some(selector) => {
                let gpu = doc.entry("gpu").or_insert_with(|| {
                    let mut t = toml_edit::Table::new();
                    t.set_implicit(true);
                    toml_edit::Item::Table(t)
                });
                let gpu = gpu.as_table_mut().context("`gpu` is not a table")?;
                gpu.entry(selector)
                    .or_insert_with(|| toml_edit::Item::Table(toml_edit::Table::new()))
                    .as_table_mut()
                    .with_context(|| format!("gpu.{selector} is not a table"))?
            }
        };
        let value = value.trim();
        if value.is_empty() {
            table.remove(leaf);
            continue;
        }
        let parsed: toml_edit::Value =
            value.parse().or_else(|_| format!("{value:?}").parse()).map_err(|e| anyhow!("{key}: {e}"))?;
        table[leaf] = toml_edit::value(parsed);
    }
    let out = doc.to_string();
    Config::parse(&out).context("the edited config is invalid; nothing was written")?;
    Ok(out)
}

fn set(args: &Args) -> Result<()> {
    if args.rest.is_empty() {
        bail!("set needs at least one KEY=VALUE\n\n{USAGE}");
    }
    let text = std::fs::read_to_string(&args.config).unwrap_or_default();
    let edited = edit_config(&text, &args.rest)?;
    // Check against the GPUs present when NVML is available.
    if let Ok(nvml) = Nvml::open() {
        Config::parse(&edited)?
            .resolve(nvml.devices())
            .context("the edited config does not fit these GPUs; nothing was written")?;
    }
    write_atomic(&args.config, edited.as_bytes())?;
    println!("wrote {}", args.config.display());
    if args.reload {
        match running_daemon() {
            // SAFETY: kill with a verified daemon pid and a valid signal number.
            Some(pid) if unsafe { libc::kill(pid, libc::SIGHUP) } == 0 => {
                println!("reloaded gpu-powerd (pid {pid}); `gpu-powerd status` shows whether it was accepted");
            }
            _ => println!("gpu-powerd is not running; the new config applies at its next start"),
        }
    }
    Ok(())
}

/// The running daemon's pid: the lock is held, and the pid recorded in it is a
/// `gpu-powerd` process running the daemon (not `--restore`, `--check` or a stale pid).
fn running_daemon() -> Option<i32> {
    if let Ok(free) = lock(None, false) {
        drop(free);
        return None;
    }
    let pid: i32 = std::fs::read_to_string(LOCK_FILE).ok()?.trim().parse().ok()?;
    if pid <= 1 {
        return None;
    }
    let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).ok()?;
    let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    let args: Vec<&[u8]> = cmdline.split(|&b| b == 0).skip(1).collect();
    let daemon = args.iter().all(|a| !matches!(*a, b"--restore" | b"restore" | b"--check" | b"check" | b"set"));
    (comm.trim() == "gpu-powerd" && daemon).then_some(pid)
}

#[cfg(test)]
mod tests {
    use super::edit_config;

    #[test]
    fn sd_notify_reaches_the_socket() {
        let dir = std::env::temp_dir().join(format!("gpu-powerd-notify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("notify.sock");
        let _ = std::fs::remove_file(&path);
        let socket = std::os::unix::net::UnixDatagram::bind(&path).unwrap();
        // SAFETY: this is the only test that touches NOTIFY_SOCKET.
        unsafe { std::env::set_var("NOTIFY_SOCKET", &path) };
        super::notify("READY=1");
        let mut buffer = [0u8; 64];
        let n = socket.recv(&mut buffer).unwrap();
        assert_eq!(&buffer[..n], b"READY=1");
        // SAFETY: as above.
        unsafe { std::env::remove_var("NOTIFY_SOCKET") };
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_edits_and_validates() {
        let text = "# keep me\nsystem_budget_w = 900\n\n[gpu.0]\nmax_w = 500\n";
        let out =
            edit_config(text, &["system_budget_w=1000".into(), "gpu.1.max_w=400".into(), "hard_c=88".into()]).unwrap();
        let pci = edit_config(text, &["gpu.E1:00.0.max_w=450".into()]).unwrap();
        assert_eq!(gpu_powerd::config::Config::parse(&pci).unwrap().gpu["E1:00.0"].max_w, Some(450.0));
        assert!(out.contains("# keep me"));
        assert!(out.contains("system_budget_w = 1000"));
        assert!(out.contains("hard_c = 88"));
        let config = gpu_powerd::config::Config::parse(&out).unwrap();
        assert_eq!(config.gpu["1"].max_w, Some(400.0));
        assert_eq!(config.gpu["0"].max_w, Some(500.0));
        let out = edit_config(&out, &["gpu.0.max_w=".into(), "system_budget_w=".into()]).unwrap();
        let config = gpu_powerd::config::Config::parse(&out).unwrap();
        assert_eq!(config.gpu["0"].max_w, None);
        assert_eq!(config.system_budget_w, None);
        assert!(edit_config(text, &["hard_c=50".into()]).is_err(), "hard below throttle start");
        assert!(edit_config(text, &["nonsense=1".into()]).is_err());
        assert!(edit_config(text, &["a.b=1".into()]).is_err());
    }
}
