# gpu-powerd

A thermal- and budget-aware power-limit daemon for NVIDIA GPUs, with a live sample
stream. Every period (1 s by default) it sets each GPU's power limit from three things:

- **A shared power budget**, split by load. An idle card stays near its base, a busy
  card gets up to its own max, and two busy cards split by demand (900 W becomes
  450/450, or 600/300 when one wants less).
- **A target temperature for each card.** A card that runs cool is never capped. A hot
  card is held just under `target_c`: until the card's learned model is confident, a
  PI controller does this; after that, the model does.
- **A learned thermal model for each card.** It predicts the card's temperature at a
  given power, including heat from the other card, and sets the highest limit whose
  predicted temperature stays at or below target. It acts *before* the card heats up,
  so neither the firmware's late, aggressive thermal throttle nor this daemon's own
  hard cut has to act.

It replaces a fixed boot-time limit (`nvidia-smi -pl` in a oneshot unit). It does not
control fans; it records fan speed.

## Why

Some boards have no firmware temperature target, only a power limit. Under sustained
load (long LLM prefills, for example), a card with poor airflow can heat until it
falls off the PCIe bus (Xid 79). A fixed low limit prevents that, but costs
performance all the time, including on the cool card and in short bursts. gpu-powerd
lets each card run at full power while it is cool and the budget allows, and holds a
hot card at the power that keeps it at `target_c`.

## How each limit is chosen

Each GPU's limit is `min(budget share, thermal limit, max_w)`, floored at `base_w`.
The thermal limit depends on the card's temperature:

| condition | thermal limit |
|---|---|
| a failed or implausible reading | `sensor_fail_w` (default `base_w`), at once |
| at or above `hard_c` (measured or smoothed) | `base_w`, at once |
| model confident (`use_model`) | the model's plan, less the measured-error correction |
| no confident model, at or above `throttle_start_c` | PI controller toward `target_c` |
| no confident model, below `throttle_start_c` | none (`max_w`) |

- **The model's plan** is the highest power for which the card's predicted temperature
  stays at or below `target_c` over `model_horizon_s`, given the other cards' current
  power and the cross-coupling. A cold card can therefore run above its steady-state
  power while it has thermal headroom to spend, and the allowance closes smoothly as it
  approaches target.
- **The measured-error correction** catches a wrong or stale model. An observer runs
  the model alongside the measured temperature and scales the model's heating by a gain
  fitted to what it measures; the plan is made with that gain. A model whose heat
  transfer is off by 2x either way is corrected this way. In simulation, the card never
  goes past 80 °C (target) before the gain settles. While the card is above target, the
  limit is also cut by `correction_kp_w_per_c` per degree.
- **The PI controller** lowers the limit while the card is above target and releases it
  while the card is below. Its integral bleeds away below `throttle_start_c`.
- **`model_may_raise = false`** limits the model to lowering what the PI controller
  would set.

## Safety rules

The safety bounds are configuration, checked as ordering invariants on every load and
reload: `throttle_start_c < target_c < hard_c`, `base_w <= fallback_w <= max_w`, and
every wattage (including `sensor_fail_w`) within the board's range. A reload that fails
validation is rejected, and the running config stays in force.

No controller can override these bounds:

- **Hard cut.** At or above `hard_c` the card drops straight to `base_w`.
- **Sensor failure.** A failed or implausible reading puts that card at `sensor_fail_w`
  until readings return. A temperature outside 0 to 125 °C, power outside 0 to 2000 W,
  and so on count as implausible.
- **Ceilings.** `max_w` and the budget are hard ceilings, and the model never plans past
  `target_c`.
- **Cut fast, raise slowly.** Every cut applies immediately. A thermal limit rises by at
  most `raise_w_per_s` (10 W/s by default); a budget share grows by at most
  `budget_raise_w_per_s`.
- **Every exit leaves a known limit.** On a stop, an error, a panic or a failed limit
  write, every GPU is set to `fallback_w`. This happens inside the process (Drop) and
  again from `ExecStopPost=gpu-powerd --restore`. The unit runs with `Restart=always`
  and a systemd watchdog, and a lock file keeps it to a single instance.
- **Light on the driver.** NVML is opened once, in-process (`dlopen` of
  `libnvidia-ml.so.1`); `nvidia-smi` is never spawned.
  - The per-sample reads (temperature, power, utilization, enforced limit, SM and memory
    clocks, clock event reasons) cost about 10 µs per GPU on an RTX PRO 6000 with driver
    595, so periods down to 0.1 s are fine.
  - Fan speed and the T.Limit margin go to the GPU's firmware (1 to 7 ms per call), so
    they are read every `fan_read_s` and `tlimit_read_s` (10 s by default).
  - VRAM (`nvmlDeviceGetMemoryInfo_v2`: total, used, free, reserved) is read every sample
    (about 4 µs per GPU; it is not a firmware RPC). The per-process list
    (`nvmlDeviceGetComputeRunningProcesses_v3`, 0.04 to 0.12 ms per GPU) is cached and
    off by default; set `process_read_s` (seconds) to refresh it. Where NVML reports
    per-process memory as N/A (WSL), the process entry has no `used_mib`.
  - A limit is written only when its quantised value changes, and rewritten every
    `reassert_s` in case another tool changed it.

The daemon senses the GPU core temperature only. Boards that do not report memory or
VRM temperature (the RTX PRO 6000 does not) are protected through the core temperature
alone.

## How the budget is split

Every GPU first gets `base_w`. Demand then comes from load:

- An idle GPU (low utilization, not power-limited) wants `base_w`.
- A power-limited GPU wants its full thermal limit. Power-limited means drawing at least
  95% of its enforced limit, or reporting a power-cap clock event while busy.
- Any other busy GPU wants its draw plus `demand_headroom_w`.

The budget above the bases is water-filled up to each GPU's demand (max-min fair: equal
shares, and a GPU that wants less leaves the rest to the others). Whatever is left over
is filled up to each GPU's thermal limit.

With `idle_takes_spare = true` (the default), idle cards share that leftover too. With a
900 W budget and one card busy at 600 W, the idle card holds 300 W rather than 150 W, so
it can start work at full speed without waiting for the budget to move. The total never
exceeds the budget either way, and the shares rebalance as soon as the idle card draws.

Demand never exceeds a GPU's thermal limit. So when a hot card is held at target, the
budget it cannot use goes to the cooler card, within that card's own thermal limit.

## The thermal model

Each GPU is modelled as first order, with self and cross coupling:

    τ_i · dT_i/dt = T_amb + Σ_j R_ij·P_j − T_i

so `T_i → T_amb + R_ii·P_i + R_ij·P_j` with time constant `τ_i`.

- **Fitting.** Readings are grouped into 5 s windows. The model is fitted online by
  recursive least squares, with a forgetting factor that applies only while the inputs
  move, so long idle periods do not erase what was learned under load.
- **When it acts.** The model takes over from the PI controller once it is confident:
  `model_min_samples` windows, `model_min_power_std_w` of power variation, a fit error
  below `model_max_residual_c`, and plausible parameters.
- **Persistence.** It is saved to `/var/lib/gpu-powerd/model.json` every `model_save_s`
  and on exit, and reloaded at start if the GPU set (by UUID) is unchanged.
- **Inspection.** `gpu-powerd model` shows R, τ, T_amb, the fit error, the confidence and
  the sample count for each GPU.

In simulation (`tests/sim.rs`), on a two-GPU plant where GPU 1 would settle near 100 °C
at full power, with target 80 °C and hard 85 °C:

| run | GPU 1 peak | mean abs(T − 80) |
|---|---|---|
| sustained load, PI alone | 83.6 °C | 0.2 °C |
| sustained load, model leading | 79.5 °C | 0.5 °C |
| model with R off by ½× | 79.5 °C | 0.5 °C |
| model with R off by 2× | 79.6 °C | 0.4 °C |
| learned online, then leading | 79.6 °C | 0.4 °C |

- **Bursts.** Over three 15-minute bursts, GPU 1 spends 1,280 °C·s above target under PI
  alone and none with the model leading.
- **Fitting accuracy.** Four hours of mixed load fit R within 10% and τ within 20%.
- **Cost of the margin.** The model holds about 0.5 °C under target, because limits are
  rounded down to the 5 W step. That costs under 3% power against PI alone.

## Live stream (WebSocket)

`gpu-powerd-ws` serves every sample as one JSON text message over WebSocket, on
`websocket_bind:websocket_port` (default `0.0.0.0:9877`).

- **Fields per GPU.** Temperature; T.Limit margin; power draw and limit; SM and memory
  clocks; GPU and memory utilization; active throttle reasons; fan %; the limit's bound
  and reason; the measured-error correction; and model status.
- **Read-only.** Client messages are ignored (only close and ping are honoured), and
  there are no commands.

> **No authentication.** Anyone who can reach the port sees the samples. They are
> readings, not secrets, but bind it to a trusted network, or to `127.0.0.1`, or turn
> it off with `websocket = false`.

The root daemon never opens a network socket. It publishes each sample on a local Unix
socket (`stream_socket`, `/run/gpu-powerd/stream.sock`), and the relay is a separate
service that runs as a transient unprivileged user (`DynamicUser=yes`), with no
capabilities and `ProtectSystem=strict`.

Any client works, for example `websocat ws://HOST:9877/`, or in a browser:

```js
new WebSocket("ws://HOST:9877/").onmessage = (e) => console.log(JSON.parse(e.data));
```

## Install (Homebrew)

```sh
brew install tpurtell/local-ai/gpu-powerd
sudo "$(brew --prefix)/bin/gpu-powerd-setup"
```

gpu-powerd runs as root, so it must not execute from the Homebrew prefix, which your
user can write. The formula builds a root bundle: both service binaries, patched to the
system loader, with no embedded library paths, packed as a tarball.
`gpu-powerd-setup` verifies the bundle and installs root-owned copies:

- `/usr/local/sbin/gpu-powerd` and `/etc/systemd/system/gpu-powerd.service` (root)
- `/usr/local/bin/gpu-powerd-ws` and `/etc/systemd/system/gpu-powerd-ws.service`
  (dynamic user)
- `/etc/gpu-powerd/config.toml` (written from the example only if absent)

It refuses to run on a host without an NVIDIA driver and GPUs. If
`nvidia-power-limit.service` is enabled, setup disables it and records that in
`/var/lib/gpu-powerd/replaced`. Run setup again after `brew upgrade gpu-powerd`.

```sh
sudo "$(brew --prefix)/bin/gpu-powerd-setup" --uninstall   # stop, remove, re-enable what it replaced
"$(brew --prefix)/bin/gpu-powerd-setup" --check            # verify the bundle (no root)
```

## Use

```sh
gpu-powerd --check            # read-only, no root: readings, board ranges, the plan, the model
gpu-powerd status             # the running daemon's limits and reasons (--json for the state file)
gpu-powerd model              # the learned model per GPU
sudo /usr/local/sbin/gpu-powerd set system_budget_w=1000 gpu.E1:00.0.target_c=78
sudo systemctl reload gpu-powerd   # after editing /etc/gpu-powerd/config.toml by hand
```

- `set` edits the config in place (comments are kept), validates the result against the
  GPUs present, and sends SIGHUP to the daemon. `KEY=` removes a setting.
- Run `set` as root from `/usr/local/sbin`. Never `sudo` the Homebrew copy.
- A rejected reload is shown by `status`.
- `gpu-powerd-ws` reads the config at start; restart it after changing its keys.

## Configuration

Every value is a config key. [`config.example.toml`](config.example.toml) lists every
key with its default, and
[`deploy/examples/two-600w-cards.toml`](deploy/examples/two-600w-cards.toml) is a
complete config for two 600 W cards sharing 900 W.

Per-card keys can be set at the top level or per card, in `[gpu."SELECTOR"]` tables. A
selector is an NVML index (`[gpu.0]`), a UUID (`[gpu."GPU-…"]`) or a PCI bus id
(`[gpu."E1:00.0"]`). The per-card keys are:

- the wattages `base_w`, `max_w`, `fallback_w` and `sensor_fail_w`;
- the temperatures `throttle_start_c`, `target_c` and `hard_c`;
- the PI and correction gains;
- `use_model` and `model_may_raise`.

## State file

`/run/gpu-powerd/state.json` is rewritten at most once a second and holds the same JSON
as the stream. For each GPU it holds:

- the applied limit, its bound (`max`, `budget`, `pi`, `model`, `correction`,
  `recovering`, `hard`, `sensor`) and a readable reason;
- the readings (including `mem_total_mib`, `mem_used_mib`, `mem_free_mib` and
  `mem_reserved_mib`; any unreadable value is null) and active throttle reasons;
- `processes`: the compute processes (`pid`, `used_mib`) as of the last `process_read_s`
  refresh, null when off or unreadable;
- the smoothed values, demand and share;
- the model summary and its corrected gain.

On exit, `running` becomes false.

## Build and test

```sh
cargo test --release      # control law, model and stream against a simulated two-GPU plant
cargo run --release --bin gpu-powerd -- --check
cargo run --release --example nvml_timing   # read-only NVML call costs on this host
```

There are no runtime dependencies: two Rust binaries, linked only against glibc. NVML
is `dlopen`ed at runtime, so there is no link-time dependency on the driver.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
