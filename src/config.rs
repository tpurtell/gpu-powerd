//! The TOML config, its validation, and resolution against the GPUs present.
use crate::backend::DeviceInfo;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const DEFAULT_CONFIG: &str = "/etc/gpu-powerd/config.toml";
pub const DEFAULT_STATE_FILE: &str = "/run/gpu-powerd/state.json";
pub const DEFAULT_MODEL_FILE: &str = "/var/lib/gpu-powerd/model.json";
pub const LOCK_FILE: &str = "/run/lock/gpu-powerd.lock";

/// Per-GPU limits; unset values inherit the top level, then the board's own range.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GpuLimitsConfig {
    pub base_w: Option<f64>,
    pub max_w: Option<f64>,
    pub fallback_w: Option<f64>,
}

/// The config file as written. Every field has a default, so an empty file is valid.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    /// Total for all GPUs; unset means no shared budget (each GPU may reach its max_w).
    pub system_budget_w: Option<f64>,
    /// No thermal capping below this (smoothed) temperature; the model's target.
    pub throttle_start_c: f64,
    /// At or above this (measured or smoothed) temperature a GPU drops straight to base.
    pub hard_c: f64,
    /// Defaults for every GPU (board minimum / maximum when unset; fallback = base).
    pub base_w: Option<f64>,
    pub max_w: Option<f64>,
    pub fallback_w: Option<f64>,
    /// Per-GPU overrides, keyed by NVML index ("0"), UUID ("GPU-…") or PCI bus id ("E1:00.0").
    pub gpu: BTreeMap<String, GpuLimitsConfig>,

    /// Control period.
    pub period_s: f64,
    /// Limits are written in multiples of this.
    pub step_w: f64,
    /// Rewrite an unchanged limit this often (another tool may have changed it).
    pub reassert_s: f64,
    /// Smoothing time constant for temperature and power.
    pub smoothing_s: f64,
    /// How fast a thermal (ramp/model) cap may rise again; cuts are immediate.
    pub raise_w_per_s: f64,
    /// How fast a budget share may grow; cuts are immediate.
    pub budget_raise_w_per_s: f64,
    /// A GPU drawing at least this fraction of its enforced limit is power-limited and wants more.
    pub pinned_fraction: f64,
    /// Below this utilization (percent) a GPU that is not power-limited counts as idle.
    pub idle_util_pct: f64,
    /// A busy GPU that is not power-limited wants its smoothed draw plus this much.
    pub demand_headroom_w: f64,

    /// Fit the thermal model online.
    pub learn: bool,
    /// Let a confident model lower limits.
    pub use_model: bool,
    /// The model keeps predicted steady state at or below throttle_start_c minus this.
    pub model_margin_c: f64,
    /// Fit window (seconds of 1 Hz samples per regression row).
    pub model_window_s: f64,
    /// Per-window forgetting factor while the inputs move (1 = never forget).
    pub model_forgetting: f64,
    /// Windows needed before the model acts.
    pub model_min_samples: u64,
    /// Power variation (weighted standard deviation, W) needed before the model acts.
    pub model_min_power_std_w: f64,
    /// Residual (RMS, °C per window) above which the model does not act.
    pub model_max_residual_c: f64,

    pub state_file: PathBuf,
    pub model_file: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            system_budget_w: None,
            throttle_start_c: 75.0,
            hard_c: 85.0,
            base_w: None,
            max_w: None,
            fallback_w: None,
            gpu: BTreeMap::new(),
            period_s: 1.0,
            step_w: 5.0,
            reassert_s: 30.0,
            smoothing_s: 3.0,
            raise_w_per_s: 10.0,
            budget_raise_w_per_s: 100.0,
            pinned_fraction: 0.95,
            idle_util_pct: 5.0,
            demand_headroom_w: 50.0,
            learn: true,
            use_model: true,
            model_margin_c: 1.0,
            model_window_s: 5.0,
            model_forgetting: 0.999,
            model_min_samples: 180,
            model_min_power_std_w: 40.0,
            model_max_residual_c: 1.5,
            state_file: DEFAULT_STATE_FILE.into(),
            model_file: DEFAULT_MODEL_FILE.into(),
        }
    }
}

/// One GPU's resolved limits.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct GpuLimits {
    pub base_w: f64,
    pub max_w: f64,
    pub fallback_w: f64,
}

/// A config resolved against the GPUs present: what the controller runs on.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub config: Config,
    pub gpus: Vec<GpuLimits>,
    /// `system_budget_w`, or the sum of every max_w when unset.
    pub budget_w: f64,
}

fn check(condition: bool, message: impl FnOnce() -> String) -> Result<()> {
    if condition { Ok(()) } else { bail!(message()) }
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

    /// Checks that need no GPU facts.
    pub fn validate(&self) -> Result<()> {
        let finite = |name: &str, v: f64| check(v.is_finite(), || format!("{name} must be a finite number"));
        for (name, v) in [
            ("throttle_start_c", self.throttle_start_c),
            ("hard_c", self.hard_c),
            ("period_s", self.period_s),
            ("step_w", self.step_w),
            ("reassert_s", self.reassert_s),
            ("smoothing_s", self.smoothing_s),
            ("raise_w_per_s", self.raise_w_per_s),
            ("budget_raise_w_per_s", self.budget_raise_w_per_s),
            ("pinned_fraction", self.pinned_fraction),
            ("idle_util_pct", self.idle_util_pct),
            ("demand_headroom_w", self.demand_headroom_w),
            ("model_margin_c", self.model_margin_c),
            ("model_window_s", self.model_window_s),
            ("model_forgetting", self.model_forgetting),
            ("model_min_power_std_w", self.model_min_power_std_w),
            ("model_max_residual_c", self.model_max_residual_c),
        ] {
            finite(name, v)?;
        }
        check((30.0..=110.0).contains(&self.throttle_start_c), || "throttle_start_c must be in 30..=110".into())?;
        check(self.hard_c > self.throttle_start_c && self.hard_c <= 120.0, || {
            "hard_c must be above throttle_start_c and at most 120".into()
        })?;
        check((0.1..=10.0).contains(&self.period_s), || "period_s must be in 0.1..=10".into())?;
        check((1.0..=50.0).contains(&self.step_w), || "step_w must be in 1..=50".into())?;
        check(self.reassert_s >= self.period_s, || "reassert_s must be at least period_s".into())?;
        check(self.smoothing_s >= 0.0 && self.smoothing_s <= 60.0, || "smoothing_s must be in 0..=60".into())?;
        check(self.raise_w_per_s > 0.0, || "raise_w_per_s must be positive".into())?;
        check(self.budget_raise_w_per_s > 0.0, || "budget_raise_w_per_s must be positive".into())?;
        check((0.5..=1.0).contains(&self.pinned_fraction), || "pinned_fraction must be in 0.5..=1".into())?;
        check((0.0..=100.0).contains(&self.idle_util_pct), || "idle_util_pct must be in 0..=100".into())?;
        check(self.demand_headroom_w >= 0.0, || "demand_headroom_w must not be negative".into())?;
        check((0.0..=20.0).contains(&self.model_margin_c), || "model_margin_c must be in 0..=20".into())?;
        check((1.0..=60.0).contains(&self.model_window_s), || "model_window_s must be in 1..=60".into())?;
        check(self.model_forgetting > 0.9 && self.model_forgetting <= 1.0, || {
            "model_forgetting must be in (0.9, 1]".into()
        })?;
        check(self.model_min_power_std_w >= 0.0, || "model_min_power_std_w must not be negative".into())?;
        check(self.model_max_residual_c > 0.0, || "model_max_residual_c must be positive".into())?;
        if let Some(budget) = self.system_budget_w {
            check(budget.is_finite() && budget > 0.0, || "system_budget_w must be positive".into())?;
        }
        let mut limits: Vec<(String, &GpuLimitsConfig)> = vec![];
        let top = GpuLimitsConfig { base_w: self.base_w, max_w: self.max_w, fallback_w: self.fallback_w };
        limits.push(("top level".into(), &top));
        for (key, gpu) in &self.gpu {
            limits.push((format!("[gpu.\"{key}\"]"), gpu));
        }
        for (place, l) in limits {
            for (name, v) in [("base_w", l.base_w), ("max_w", l.max_w), ("fallback_w", l.fallback_w)] {
                if let Some(v) = v {
                    check(v.is_finite() && v > 0.0 && v <= 2000.0, || format!("{place}: {name} must be in (0, 2000]"))?;
                }
            }
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
        let mut gpus = Vec::with_capacity(devices.len());
        for (i, device) in devices.iter().enumerate() {
            let own = matched[i].map(|key| &self.gpu[key]);
            let pick = |own: Option<f64>, top: Option<f64>| own.or(top);
            let base_w = pick(own.and_then(|g| g.base_w), self.base_w).unwrap_or(device.board_min_w);
            let max_w = pick(own.and_then(|g| g.max_w), self.max_w).unwrap_or(device.board_max_w);
            let fallback_w = pick(own.and_then(|g| g.fallback_w), self.fallback_w).unwrap_or(base_w);
            let name = format!("GPU {i} ({})", device.uuid);
            let (low, high) = (device.board_min_w, device.board_max_w);
            check(base_w <= max_w, || format!("{name}: base_w {base_w} exceeds max_w {max_w}"))?;
            for (key, v) in [("base_w", base_w), ("max_w", max_w), ("fallback_w", fallback_w)] {
                check(v >= low - 0.5 && v <= high + 0.5, || {
                    format!("{name}: {key} {v} W is outside the board range {low}..{high} W")
                })?;
            }
            gpus.push(GpuLimits { base_w, max_w, fallback_w });
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
    !pci.is_empty() && (pci == key || pci.ends_with(&format!(":{key}")) || pci.ends_with(&key) && key.contains(':'))
}

#[cfg(test)]
mod tests {
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
    fn empty_config_uses_board_range() {
        let settings = Config::parse("").unwrap().resolve(&devices()).unwrap();
        assert_eq!(settings.gpus[0], GpuLimits { base_w: 150.0, max_w: 600.0, fallback_w: 150.0 });
        assert_eq!(settings.budget_w, 1200.0);
    }

    #[test]
    fn per_gpu_overrides_by_index_uuid_and_pci() {
        let text = r#"
            system_budget_w = 900
            max_w = 500
            fallback_w = 325
            [gpu."E1:00.0"]
            max_w = 400
            [gpu.0]
            base_w = 200
        "#;
        let settings = Config::parse(text).unwrap().resolve(&devices()).unwrap();
        assert_eq!(settings.gpus[0], GpuLimits { base_w: 200.0, max_w: 500.0, fallback_w: 325.0 });
        assert_eq!(settings.gpus[1], GpuLimits { base_w: 150.0, max_w: 400.0, fallback_w: 325.0 });
        let uuid = "[gpu.\"GPU-00000001-aaaa-bbbb-cccc-dddddddddddd\"]\nmax_w = 300\n";
        let settings = Config::parse(uuid).unwrap().resolve(&devices()).unwrap();
        assert_eq!(settings.gpus[1].max_w, 300.0);
    }

    #[test]
    fn rejects_bad_configs() {
        assert!(Config::parse("hard_c = 70").is_err(), "hard below throttle start");
        assert!(Config::parse("unknown_key = 1").is_err());
        assert!(Config::parse("[gpu.0]\nmax = 1").is_err());
        let d = devices();
        assert!(Config::parse("max_w = 700").unwrap().resolve(&d).is_err(), "above board max");
        assert!(Config::parse("system_budget_w = 250").unwrap().resolve(&d).is_err(), "below base sum");
        assert!(Config::parse("[gpu.7]\nmax_w = 300").unwrap().resolve(&d).is_err(), "no such GPU");
        assert!(Config::parse("base_w = 400\nmax_w = 300").unwrap().resolve(&d).is_err());
    }

    #[test]
    fn example_config_parses() {
        let text = include_str!("../config.example.toml");
        Config::parse(text).unwrap().resolve(&devices()).unwrap();
    }
}
