//! The TOML config: every tunable, its default (in one place, [`Config::default`] and
//! [`CardConfig::default`]), validation, and resolution against the GPUs present.
//!
//! Per-card keys may be set at the top level (every GPU) and overridden in
//! `[gpu."SELECTOR"]` tables (NVML index, UUID or PCI bus id). The safety rules are
//! ordering invariants checked on every load and reload:
//! `throttle_start_c < target_c < hard_c`, `base_w <= fallback_w <= max_w`, and every
//! wattage within the board's range.
use crate::backend::DeviceInfo;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const DEFAULT_CONFIG: &str = "/etc/gpu-powerd/config.toml";
pub const DEFAULT_STATE_FILE: &str = "/run/gpu-powerd/state.json";
pub const DEFAULT_MODEL_FILE: &str = "/var/lib/gpu-powerd/model.json";
pub const DEFAULT_STREAM_SOCKET: &str = "/run/gpu-powerd/stream.sock";
/// In the daemon's own root-owned runtime directory (systemd RuntimeDirectory, 0755): an
/// unprivileged process can never create it, unlike a file in sticky world-writable /run/lock,
/// where a user-owned lock would lock root out (fs.protected_regular).
pub const LOCK_FILE: &str = "/run/gpu-powerd/daemon.lock";

/// Keys that may differ per card. `None` inherits: the top level, then the default.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CardOverrides {
    pub base_w: Option<f64>,
    pub max_w: Option<f64>,
    pub fallback_w: Option<f64>,
    pub sensor_fail_w: Option<f64>,
    pub throttle_start_c: Option<f64>,
    pub target_c: Option<f64>,
    pub hard_c: Option<f64>,
    pub pi_kp_w_per_c: Option<f64>,
    pub pi_ki_w_per_c_s: Option<f64>,
    pub correction_kp_w_per_c: Option<f64>,
    pub correction_rate_per_s: Option<f64>,
    pub use_model: Option<bool>,
    pub model_may_raise: Option<bool>,
}

/// One card's resolved settings.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct CardConfig {
    /// The lowest limit the daemon sets (board minimum when unset).
    pub base_w: f64,
    /// The highest limit the daemon sets (board maximum when unset).
    pub max_w: f64,
    /// The limit left behind on exit, crash or a failed write (base_w when unset).
    pub fallback_w: f64,
    /// The limit while this card's readings fail or are implausible (base_w when unset).
    pub sensor_fail_w: f64,
    /// Below this (smoothed) temperature no thermal limit applies at all.
    pub throttle_start_c: f64,
    /// The temperature the controller holds a hot card at.
    pub target_c: f64,
    /// At or above this (measured or smoothed) temperature: straight to base_w.
    pub hard_c: f64,
    /// PI controller (before the model is confident): watts per °C of error.
    pub pi_kp_w_per_c: f64,
    /// PI controller: watts per °C·s of accumulated error.
    pub pi_ki_w_per_c_s: f64,
    /// Measured-error correction on the model: watts cut per °C above target_c.
    pub correction_kp_w_per_c: f64,
    /// How fast the model's heating gain adapts to the measured error (fraction of the
    /// error per second; four times faster when the card runs hotter than predicted).
    pub correction_rate_per_s: f64,
    /// Let a confident model set the thermal limit.
    pub use_model: bool,
    /// Let the model raise the thermal limit above what the PI controller would give
    /// (never past max_w, the budget, or a predicted target_c).
    pub model_may_raise: bool,
}

/// Top-level card defaults: the one place every per-card default lives.
fn card_defaults() -> CardOverrides {
    CardOverrides {
        base_w: None,
        max_w: None,
        fallback_w: None,
        sensor_fail_w: None,
        throttle_start_c: Some(75.0),
        target_c: Some(80.0),
        hard_c: Some(85.0),
        pi_kp_w_per_c: Some(25.0),
        pi_ki_w_per_c_s: Some(1.0),
        correction_kp_w_per_c: Some(25.0),
        correction_rate_per_s: Some(0.05),
        use_model: Some(true),
        model_may_raise: Some(true),
    }
}

/// The config file as written. Every field has a default, so an empty file is valid.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    /// Total for all GPUs; unset means no shared budget (each GPU may reach its max_w).
    pub system_budget_w: Option<f64>,
    /// Per-card keys for every card.
    #[serde(flatten)]
    pub card: CardOverrides,
    /// Per-card overrides, keyed by NVML index ("0"), UUID ("GPU-…") or PCI bus id ("E1:00.0").
    pub gpu: BTreeMap<String, CardOverrides>,

    /// Sample and control period (seconds); down to 0.1 for a fast stream.
    pub period_s: f64,
    /// Fan speed is a firmware RPC (ms per fan): read it this often.
    pub fan_read_s: f64,
    /// The T.Limit margin is a firmware RPC: read it this often; 0 = never.
    pub tlimit_read_s: f64,
    /// The per-process VRAM list is cheap but not free: refresh it this often; 0 = never.
    pub process_read_s: f64,
    /// Limits are written in multiples of this.
    pub step_w: f64,
    /// Rewrite an unchanged limit this often (another tool may have changed it).
    pub reassert_s: f64,
    /// Smoothing time constant for temperature and power.
    pub smoothing_s: f64,
    /// How fast a thermal limit may rise; cuts are immediate.
    pub raise_w_per_s: f64,
    /// How fast a budget share may grow; cuts are immediate.
    pub budget_raise_w_per_s: f64,
    /// A GPU drawing at least this fraction of its enforced limit is power-limited and wants more.
    pub pinned_fraction: f64,
    /// Below this utilization (percent) a GPU that is not power-limited counts as idle.
    pub idle_util_pct: f64,
    /// A busy GPU that is not power-limited wants its smoothed draw plus this much.
    pub demand_headroom_w: f64,
    /// An idle GPU also takes a share of budget the busy GPUs do not want (false: base only).
    pub idle_takes_spare: bool,

    /// Fit the thermal model online.
    pub learn: bool,
    /// The model plans so the predicted temperature over this horizon (and at steady
    /// state) stays at or below target_c; 0 = steady state only.
    pub model_horizon_s: f64,
    /// Fit window (seconds of samples per regression row).
    pub model_window_s: f64,
    /// Per-window forgetting factor while the inputs move (1 = never forget).
    pub model_forgetting: f64,
    /// Windows needed before the model acts.
    pub model_min_samples: u64,
    /// Power variation (weighted standard deviation, W) needed before the model acts.
    pub model_min_power_std_w: f64,
    /// Residual (RMS, °C per window) above which the model does not act.
    pub model_max_residual_c: f64,
    /// Save the learned model this often (and on exit).
    pub model_save_s: f64,

    pub state_file: PathBuf,
    pub model_file: PathBuf,

    /// Publish every sample to `stream_socket` for `gpu-powerd-ws`.
    pub stream: bool,
    /// The local Unix socket the root daemon publishes samples on (no network socket).
    pub stream_socket: PathBuf,
    /// `gpu-powerd-ws`: serve the stream over WebSocket (read-only, no authentication).
    pub websocket: bool,
    pub websocket_bind: String,
    pub websocket_port: u16,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            system_budget_w: None,
            card: card_defaults(),
            gpu: BTreeMap::new(),
            period_s: 1.0,
            fan_read_s: 10.0,
            tlimit_read_s: 10.0,
            process_read_s: 0.0,
            step_w: 5.0,
            reassert_s: 30.0,
            smoothing_s: 3.0,
            raise_w_per_s: 10.0,
            budget_raise_w_per_s: 100.0,
            pinned_fraction: 0.95,
            idle_util_pct: 5.0,
            demand_headroom_w: 50.0,
            idle_takes_spare: true,
            learn: true,
            model_horizon_s: 120.0,
            model_window_s: 5.0,
            model_forgetting: 0.999,
            model_min_samples: 180,
            model_min_power_std_w: 40.0,
            model_max_residual_c: 1.5,
            model_save_s: 300.0,
            state_file: DEFAULT_STATE_FILE.into(),
            model_file: DEFAULT_MODEL_FILE.into(),
            stream: true,
            stream_socket: DEFAULT_STREAM_SOCKET.into(),
            websocket: true,
            websocket_bind: "0.0.0.0".into(),
            websocket_port: 9877,
        }
    }
}

/// A config resolved against the GPUs present: what the controller runs on.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub config: Config,
    pub gpus: Vec<CardConfig>,
    /// `system_budget_w`, or the sum of every max_w when unset.
    pub budget_w: f64,
}

fn check(condition: bool, message: impl FnOnce() -> String) -> Result<()> {
    if condition { Ok(()) } else { bail!(message()) }
}

fn in_range(name: &str, v: f64, low: f64, high: f64) -> Result<()> {
    check(v.is_finite() && v >= low && v <= high, || format!("{name} = {v} must be in {low}..={high}"))
}

/// Merge `over` onto `under` (fields set in `over` win).
fn merge(under: &CardOverrides, over: &CardOverrides) -> CardOverrides {
    macro_rules! pick {
        ($($f:ident),*) => { CardOverrides { $($f: over.$f.or(under.$f)),* } };
    }
    pick!(
        base_w,
        max_w,
        fallback_w,
        sensor_fail_w,
        throttle_start_c,
        target_c,
        hard_c,
        pi_kp_w_per_c,
        pi_ki_w_per_c_s,
        correction_kp_w_per_c,
        correction_rate_per_s,
        use_model,
        model_may_raise
    )
}

/// Check one card's resolved values; `board` is the board range when known.
fn check_card(name: &str, c: &CardConfig, board: Option<(f64, f64)>) -> Result<()> {
    for (key, v) in [("throttle_start_c", c.throttle_start_c), ("target_c", c.target_c), ("hard_c", c.hard_c)] {
        in_range(&format!("{name}: {key}"), v, 20.0, 120.0)?;
    }
    check(c.throttle_start_c < c.target_c && c.target_c < c.hard_c, || {
        format!(
            "{name}: need throttle_start_c < target_c < hard_c (got {} < {} < {})",
            c.throttle_start_c, c.target_c, c.hard_c
        )
    })?;
    for (key, v) in [
        ("pi_kp_w_per_c", c.pi_kp_w_per_c),
        ("pi_ki_w_per_c_s", c.pi_ki_w_per_c_s),
        ("correction_kp_w_per_c", c.correction_kp_w_per_c),
    ] {
        in_range(&format!("{name}: {key}"), v, 0.0, 1000.0)?;
    }
    in_range(&format!("{name}: correction_rate_per_s"), c.correction_rate_per_s, 0.0, 1.0)?;
    for (key, v) in
        [("base_w", c.base_w), ("max_w", c.max_w), ("fallback_w", c.fallback_w), ("sensor_fail_w", c.sensor_fail_w)]
    {
        in_range(&format!("{name}: {key}"), v, 1.0, 2000.0)?;
        if let Some((low, high)) = board {
            check(v >= low - 0.5 && v <= high + 0.5, || {
                format!("{name}: {key} {v} W is outside the board range {low}..{high} W")
            })?;
        }
    }
    check(c.base_w <= c.fallback_w && c.fallback_w <= c.max_w, || {
        format!("{name}: need base_w <= fallback_w <= max_w (got {} <= {} <= {})", c.base_w, c.fallback_w, c.max_w)
    })?;
    Ok(())
}

impl Config {
    pub fn parse(text: &str) -> Result<Self> {
        let config: Self = toml::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("config {}", path.display()))
    }

    /// The top-level card keys merged over the defaults (an unset top-level key in a
    /// file falls back to the default, not to nothing).
    fn top(&self) -> CardOverrides {
        merge(&card_defaults(), &self.card)
    }

    /// Checks that need no GPU facts.
    pub fn validate(&self) -> Result<()> {
        in_range("period_s", self.period_s, 0.1, 10.0)?;
        in_range("fan_read_s", self.fan_read_s, self.period_s, 3600.0)?;
        if self.tlimit_read_s != 0.0 {
            in_range("tlimit_read_s", self.tlimit_read_s, self.period_s.max(1.0), 3600.0)?;
        }
        if self.process_read_s != 0.0 {
            in_range("process_read_s", self.process_read_s, self.period_s, 3600.0)?;
        }
        in_range("step_w", self.step_w, 1.0, 50.0)?;
        in_range("reassert_s", self.reassert_s, self.period_s, 3600.0)?;
        in_range("smoothing_s", self.smoothing_s, 0.0, 60.0)?;
        in_range("raise_w_per_s", self.raise_w_per_s, 0.1, 1000.0)?;
        in_range("budget_raise_w_per_s", self.budget_raise_w_per_s, 0.1, 10000.0)?;
        in_range("pinned_fraction", self.pinned_fraction, 0.5, 1.0)?;
        in_range("idle_util_pct", self.idle_util_pct, 0.0, 100.0)?;
        in_range("demand_headroom_w", self.demand_headroom_w, 0.0, 1000.0)?;
        in_range("model_horizon_s", self.model_horizon_s, 0.0, 3600.0)?;
        in_range("model_window_s", self.model_window_s, 1.0, 60.0)?;
        check(self.model_window_s >= self.period_s, || "model_window_s must be at least period_s".into())?;
        check(self.model_forgetting > 0.9 && self.model_forgetting <= 1.0, || {
            "model_forgetting must be in (0.9, 1]".into()
        })?;
        in_range("model_min_power_std_w", self.model_min_power_std_w, 0.0, 1000.0)?;
        in_range("model_max_residual_c", self.model_max_residual_c, 0.01, 50.0)?;
        in_range("model_save_s", self.model_save_s, 10.0, 86400.0)?;
        if let Some(budget) = self.system_budget_w {
            in_range("system_budget_w", budget, 1.0, 100_000.0)?;
        }
        check(!self.websocket_bind.trim().is_empty(), || "websocket_bind must not be empty".into())?;
        check(self.websocket_bind.parse::<std::net::IpAddr>().is_ok(), || {
            format!("websocket_bind {:?} must be an IP address", self.websocket_bind)
        })?;
        check(self.websocket_port != 0, || "websocket_port must not be 0".into())?;
        // Card values with no board facts: everything but the board range.
        let top = self.top();
        let tables = std::iter::once(("top level".to_string(), top.clone()))
            .chain(self.gpu.iter().map(|(k, o)| (format!("[gpu.\"{k}\"]"), merge(&top, o))));
        for (name, card) in tables {
            let board = (card.base_w.unwrap_or(1.0), card.max_w.unwrap_or(2000.0));
            check_card(&name, &finish(&card, board), None)?;
        }
        Ok(())
    }

    /// Resolve against the GPUs present; checks board ranges and the budget.
    pub fn resolve(&self, devices: &[DeviceInfo]) -> Result<Settings> {
        self.validate()?;
        let mut matched = vec![None::<&String>; devices.len()];
        for key in self.gpu.keys() {
            let hits: Vec<usize> = (0..devices.len()).filter(|&i| selector_matches(key, &devices[i])).collect();
            check(!hits.is_empty(), || format!("[gpu.\"{key}\"] matches no GPU (use an index, UUID or PCI bus id)"))?;
            for i in hits {
                if let Some(previous) = matched[i] {
                    bail!("GPU {i} is named by both [gpu.\"{previous}\"] and [gpu.\"{key}\"]");
                }
                matched[i] = Some(key);
            }
        }
        let top = self.top();
        let mut gpus = Vec::with_capacity(devices.len());
        for (i, device) in devices.iter().enumerate() {
            let merged = match matched[i] {
                Some(key) => merge(&top, &self.gpu[key]),
                None => top.clone(),
            };
            let board = (device.board_min_w, device.board_max_w);
            let card = finish(&merged, board);
            check_card(&format!("GPU {i} ({})", device.uuid), &card, Some(board))?;
            gpus.push(card);
        }
        let base_sum: f64 = gpus.iter().map(|g| g.base_w).sum();
        let budget_w = match self.system_budget_w {
            Some(budget) => {
                check(budget >= base_sum, || {
                    format!("system_budget_w {budget} W is below the sum of base_w ({base_sum} W)")
                })?;
                budget
            }
            None => gpus.iter().map(|g| g.max_w).sum(),
        };
        Ok(Settings { config: self.clone(), gpus, budget_w })
    }
}

/// Fill a card's unset values: wattages from the board range, the rest from the defaults.
fn finish(c: &CardOverrides, (board_min, board_max): (f64, f64)) -> CardConfig {
    let d = card_defaults();
    let base_w = c.base_w.unwrap_or(board_min);
    CardConfig {
        base_w,
        max_w: c.max_w.unwrap_or(board_max),
        fallback_w: c.fallback_w.unwrap_or(base_w),
        sensor_fail_w: c.sensor_fail_w.unwrap_or(base_w),
        throttle_start_c: c.throttle_start_c.or(d.throttle_start_c).unwrap_or_default(),
        target_c: c.target_c.or(d.target_c).unwrap_or_default(),
        hard_c: c.hard_c.or(d.hard_c).unwrap_or_default(),
        pi_kp_w_per_c: c.pi_kp_w_per_c.or(d.pi_kp_w_per_c).unwrap_or_default(),
        pi_ki_w_per_c_s: c.pi_ki_w_per_c_s.or(d.pi_ki_w_per_c_s).unwrap_or_default(),
        correction_kp_w_per_c: c.correction_kp_w_per_c.or(d.correction_kp_w_per_c).unwrap_or_default(),
        correction_rate_per_s: c.correction_rate_per_s.or(d.correction_rate_per_s).unwrap_or_default(),
        use_model: c.use_model.or(d.use_model).unwrap_or(true),
        model_may_raise: c.model_may_raise.or(d.model_may_raise).unwrap_or(true),
    }
}

/// `"0"` (NVML index), a full UUID, or a PCI bus id (full or without the domain).
pub fn selector_matches(key: &str, device: &DeviceInfo) -> bool {
    if let Ok(index) = key.parse::<u32>() {
        return index == device.index;
    }
    if key.starts_with("GPU-") {
        return key.eq_ignore_ascii_case(&device.uuid);
    }
    let pci = device.pci.to_ascii_lowercase();
    let key = key.to_ascii_lowercase();
    !pci.is_empty() && (pci == key || pci.ends_with(&format!(":{key}")))
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub fn devices() -> Vec<DeviceInfo> {
        (0..2)
            .map(|i| DeviceInfo {
                index: i,
                uuid: format!("GPU-0000000{i}-aaaa-bbbb-cccc-dddddddddddd"),
                name: "Test GPU".into(),
                pci: if i == 0 { "00000000:11:00.0".into() } else { "00000000:E1:00.0".into() },
                board_min_w: 150.0,
                board_max_w: 600.0,
                default_w: Some(600.0),
            })
            .collect()
    }

    #[test]
    fn empty_config_uses_board_range_and_defaults() {
        let settings = Config::parse("").unwrap().resolve(&devices()).unwrap();
        let g = &settings.gpus[0];
        assert_eq!((g.base_w, g.max_w, g.fallback_w, g.sensor_fail_w), (150.0, 600.0, 150.0, 150.0));
        assert_eq!((g.throttle_start_c, g.target_c, g.hard_c), (75.0, 80.0, 85.0));
        assert_eq!(settings.budget_w, 1200.0);
        assert_eq!(settings.config.websocket_port, 9877);
    }

    #[test]
    fn per_gpu_overrides_by_index_uuid_and_pci() {
        let text = r#"
            system_budget_w = 900
            max_w = 500
            fallback_w = 325
            target_c = 78
            [gpu."E1:00.0"]
            max_w = 400
            hard_c = 83
            model_may_raise = false
            [gpu.0]
            base_w = 200
            sensor_fail_w = 250
        "#;
        let settings = Config::parse(text).unwrap().resolve(&devices()).unwrap();
        let (g0, g1) = (&settings.gpus[0], &settings.gpus[1]);
        assert_eq!((g0.base_w, g0.max_w, g0.fallback_w, g0.sensor_fail_w), (200.0, 500.0, 325.0, 250.0));
        assert_eq!((g1.base_w, g1.max_w, g1.fallback_w, g1.sensor_fail_w), (150.0, 400.0, 325.0, 150.0));
        assert_eq!((g0.target_c, g0.hard_c, g1.target_c, g1.hard_c), (78.0, 85.0, 78.0, 83.0));
        assert!(g0.model_may_raise && !g1.model_may_raise);
        let uuid = "[gpu.\"GPU-00000001-aaaa-bbbb-cccc-dddddddddddd\"]\nmax_w = 300\n";
        assert_eq!(Config::parse(uuid).unwrap().resolve(&devices()).unwrap().gpus[1].max_w, 300.0);
    }

    #[test]
    fn rejects_bad_configs() {
        for bad in [
            "hard_c = 70",
            "target_c = 75",
            "target_c = 85",
            "throttle_start_c = 82",
            "[gpu.1]\ntarget_c = 90",
            "unknown_key = 1",
            "[gpu.0]\nmax = 1",
            "period_s = 0.05",
            "fan_read_s = 0.5\nperiod_s = 1",
            "process_read_s = 0.5\nperiod_s = 1",
            "process_read_s = -1",
            "websocket_bind = \"everywhere\"",
            "websocket_port = 0",
            "fallback_w = 100\nbase_w = 150",
            "pi_kp_w_per_c = -1",
        ] {
            assert!(Config::parse(bad).is_err(), "{bad:?} should be rejected");
        }
        let d = devices();
        for bad in [
            "max_w = 700",
            "system_budget_w = 250",
            "[gpu.7]\nmax_w = 300",
            "base_w = 400\nmax_w = 300",
            "fallback_w = 650",
            "max_w = 300\nfallback_w = 325",
            "sensor_fail_w = 100",
        ] {
            assert!(Config::parse(bad).and_then(|c| c.resolve(&d)).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn example_configs_parse() {
        let text = include_str!("../config.example.toml");
        let example = Config::parse(text).unwrap().resolve(&devices()).unwrap();
        let defaults = Config::default().resolve(&devices()).unwrap();
        assert_eq!(example.gpus, defaults.gpus, "the example shows every default, all commented out");
        assert_eq!(example.budget_w, defaults.budget_w);
        let pair =
            Config::parse(include_str!("../deploy/examples/two-600w-cards.toml")).unwrap().resolve(&devices()).unwrap();
        assert_eq!(pair.budget_w, 900.0);
        for g in &pair.gpus {
            assert_eq!((g.base_w, g.max_w, g.fallback_w, g.sensor_fail_w), (150.0, 600.0, 325.0, 150.0));
            assert_eq!((g.throttle_start_c, g.target_c, g.hard_c), (75.0, 80.0, 85.0));
        }
        let c = &pair.config;
        assert!(c.learn && c.websocket && c.stream && pair.gpus.iter().all(|g| g.use_model));
    }

    #[test]
    fn example_mentions_every_key() {
        let text = include_str!("../config.example.toml");
        let defaults = toml::to_string(&Config::default()).unwrap();
        let mut keys: Vec<String> =
            defaults.lines().filter_map(|l| l.split_once(" = ").map(|(k, _)| k.trim().to_string())).collect();
        keys.extend(
            ["base_w", "max_w", "fallback_w", "sensor_fail_w", "system_budget_w"].iter().map(|k| k.to_string()),
        );
        for key in keys {
            assert!(text.contains(&format!("#{key} = ")), "config.example.toml lacks `#{key} = `");
        }
    }
}
