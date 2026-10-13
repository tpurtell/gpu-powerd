//! The daemon: one controller period at a time over any [`Backend`], with limit
//! writes, learning, model persistence, the state file, and the failsafes.
use crate::backend::{Backend, DeviceInfo, Reading};
use crate::config::{Config, Settings};
use crate::control::{Bound, Controller, GpuPlan, quantise};
use crate::model::{FitSettings, ModelSummary, ThermalModel, WindowAccumulator};
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// How often the learned models are saved while running (and always on exit).
pub const MODEL_SAVE_S: f64 = 300.0;

/// Persisted models, keyed to the exact GPU set they were fitted on.
#[derive(Serialize, Deserialize)]
pub struct ModelFile {
    pub version: u32,
    pub uuids: Vec<String>,
    pub models: Vec<ThermalModel>,
}

pub fn fit_settings(config: &Config) -> FitSettings {
    FitSettings {
        window_s: config.model_window_s,
        forgetting: config.model_forgetting,
        min_samples: config.model_min_samples,
        min_power_std_w: config.model_min_power_std_w,
        max_residual_c: config.model_max_residual_c,
    }
}

pub fn load_models(path: &Path, devices: &[DeviceInfo], window_s: f64) -> Option<Vec<ThermalModel>> {
    let text = std::fs::read_to_string(path).ok()?;
    let file: ModelFile = serde_json::from_str(&text).ok()?;
    let uuids: Vec<&str> = devices.iter().map(|d| d.uuid.as_str()).collect();
    let n = devices.len();
    (file.version == 1
        && file.uuids.iter().map(String::as_str).eq(uuids)
        && file.models.len() == n
        && file.models.iter().enumerate().all(|(i, m)| m.compatible(i, n, window_s)))
    .then_some(file.models)
}

/// Write `bytes` to `path` atomically (temporary file, then rename), world-readable.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    let temporary = path.with_extension("tmp");
    let mut file = std::fs::File::create(&temporary).with_context(|| format!("create {}", temporary.display()))?;
    file.write_all(bytes)?;
    file.set_permissions(std::fs::Permissions::from_mode(0o644))?;
    drop(file);
    std::fs::rename(&temporary, path).with_context(|| format!("rename to {}", path.display()))?;
    Ok(())
}

fn enabled(path: &Path) -> bool {
    !path.as_os_str().is_empty()
}

/// One GPU's line in the state file.
#[derive(Clone, Debug, Serialize)]
pub struct GpuState {
    pub index: u32,
    pub uuid: String,
    pub name: String,
    pub pci: String,
    /// The limit last written (watts).
    pub limit_w: Option<u32>,
    pub base_w: f64,
    pub max_w: f64,
    pub fallback_w: f64,
    pub reading: Option<Reading>,
    pub plan: GpuPlan,
    pub model: ModelSummary,
    /// The model is allowed to lower this GPU's limit now.
    pub model_active: bool,
}

/// The state file's contents.
#[derive(Clone, Debug, Serialize)]
pub struct State {
    pub version: u32,
    pub updated: f64,
    pub host: String,
    pub pid: u32,
    pub running: bool,
    pub budget_w: f64,
    pub throttle_start_c: f64,
    pub hard_c: f64,
    /// Every GPU not at its max, with why; `at max` when none.
    pub reason: String,
    /// The last rejected config reload, if any.
    pub config_error: Option<String>,
    pub gpus: Vec<GpuState>,
}

pub fn now_unix() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

pub fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname").map(|h| h.trim().to_string()).unwrap_or_default()
}

/// The controller around a backend. Generic so the simulator runs the same code.
pub struct Daemon<B: Backend> {
    pub backend: B,
    pub settings: Settings,
    controller: Controller,
    models: Vec<ThermalModel>,
    windows: Vec<WindowAccumulator>,
    applied: Vec<Option<u32>>,
    applied_at: Vec<f64>,
    /// Seconds since start (sum of `dt`).
    clock: f64,
    saved_at: f64,
    pub config_error: Option<String>,
    pub last: Option<State>,
    /// Set once every GPU has been put at its fallback; Drop does nothing more.
    restored: bool,
}

impl<B: Backend> Daemon<B> {
    pub fn new(backend: B, settings: Settings) -> Self {
        let n = backend.devices().len();
        let window = settings.config.model_window_s;
        let models = if enabled(&settings.config.model_file) {
            load_models(&settings.config.model_file, backend.devices(), window)
        } else {
            None
        }
        .unwrap_or_else(|| (0..n).map(|i| ThermalModel::new(i, n, window)).collect());
        Self {
            backend,
            settings,
            controller: Controller::new(n),
            models,
            windows: vec![WindowAccumulator::default(); n],
            applied: vec![None; n],
            applied_at: vec![f64::NEG_INFINITY; n],
            clock: 0.0,
            saved_at: 0.0,
            config_error: None,
            last: None,
            restored: false,
        }
    }

    pub fn models(&self) -> &[ThermalModel] {
        &self.models
    }

    /// Replace the learned models (tests and offline tools).
    pub fn set_models(&mut self, models: Vec<ThermalModel>) {
        assert_eq!(models.len(), self.models.len());
        self.models = models;
    }

    pub fn applied(&self) -> &[Option<u32>] {
        &self.applied
    }

    /// Replace the settings (a reload). Rates carry over: a lower max cuts at once,
    /// a higher one is reached at the configured raise rates.
    pub fn reload(&mut self, settings: Settings) {
        let n = self.models.len();
        let window = settings.config.model_window_s;
        if (window - self.settings.config.model_window_s).abs() > 1e-9 {
            self.models = (0..n).map(|i| ThermalModel::new(i, n, window)).collect();
            self.windows = vec![WindowAccumulator::default(); n];
        }
        self.settings = settings;
        self.config_error = None;
        // Rewrite every limit on the next period.
        self.applied_at.iter_mut().for_each(|t| *t = f64::NEG_INFINITY);
    }

    /// One period of `dt` seconds. An error means a limit write failed: the caller
    /// must stop (dropping the daemon restores every GPU to its fallback).
    pub fn tick(&mut self, dt: f64) -> Result<&State> {
        self.clock += dt;
        let n = self.models.len();
        let readings: Vec<Result<Reading, String>> =
            (0..n).map(|i| self.backend.read(i).map_err(|e| e.to_string())).collect();
        let fit = fit_settings(&self.settings.config);
        let c = &self.settings.config;
        let summaries: Vec<ModelSummary> = self.models.iter().map(|m| m.summary(&fit)).collect();
        let active: Vec<bool> = summaries.iter().map(|s| c.use_model && s.confident).collect();
        let usable: Vec<Option<&ThermalModel>> =
            self.models.iter().zip(&active).map(|(m, &a)| a.then_some(m)).collect();
        let plans = self.controller.step(&self.settings, &readings, &usable, dt);

        // Write limits that changed (quantised), and reassert the rest periodically.
        for (i, plan) in plans.iter().enumerate() {
            let limits = &self.settings.gpus[i];
            let watts = quantise(plan.limit_w, c.step_w, limits.base_w, limits.max_w);
            let due = self.clock - self.applied_at[i] >= c.reassert_s - 1e-9;
            if self.applied[i] != Some(watts) || due {
                if self.applied[i] != Some(watts) {
                    eprintln!(
                        "gpu-powerd: GPU {i} limit {} -> {watts} W ({})",
                        self.applied[i].map_or("?".into(), |w| w.to_string()),
                        plan.reason
                    );
                }
                self.backend.set_limit_w(i, watts).map_err(|e| anyhow!("GPU {i}: setting {watts} W failed: {e}"))?;
                self.applied[i] = Some(watts);
                self.applied_at[i] = self.clock;
            }
        }

        // Learning: every GPU's reading must be good for a sample to count.
        if c.learn {
            let all: Option<Vec<&Reading>> = readings.iter().map(|r| r.as_ref().ok()).collect();
            let all = all.filter(|rs| rs.iter().all(|r| crate::control::plausible(r).is_ok()));
            let powers: Vec<f64> = all.as_ref().map(|rs| rs.iter().map(|r| r.power_w).collect()).unwrap_or_default();
            for i in 0..n {
                let sample = all.as_ref().map(|rs| (rs[i].temperature_c, powers.as_slice(), rs[i].fan_pct));
                if let Some(w) = self.windows[i].push(sample, dt, c.model_window_s) {
                    self.models[i].update(w.t_prev, w.t_next, &w.powers, w.fan_pct, &fit);
                }
            }
            if self.clock - self.saved_at >= MODEL_SAVE_S {
                self.saved_at = self.clock;
                if let Err(e) = self.save_models() {
                    eprintln!("gpu-powerd: saving the model failed: {e:#}");
                }
            }
        }

        let summaries: Vec<ModelSummary> = self.models.iter().map(|m| m.summary(&fit)).collect();
        let devices = self.backend.devices();
        let gpus: Vec<GpuState> = (0..n)
            .map(|i| GpuState {
                index: devices[i].index,
                uuid: devices[i].uuid.clone(),
                name: devices[i].name.clone(),
                pci: devices[i].pci.clone(),
                limit_w: self.applied[i],
                base_w: self.settings.gpus[i].base_w,
                max_w: self.settings.gpus[i].max_w,
                fallback_w: self.settings.gpus[i].fallback_w,
                reading: readings[i].clone().ok(),
                plan: plans[i].clone(),
                model: summaries[i].clone(),
                model_active: active[i],
            })
            .collect();
        let reason = gpus
            .iter()
            .filter(|g| g.plan.bound != Bound::Max)
            .map(|g| format!("GPU {} {} W: {}", g.index, g.limit_w.unwrap_or(0), g.plan.reason))
            .collect::<Vec<_>>()
            .join("; ");
        let state = State {
            version: 1,
            updated: now_unix(),
            host: hostname(),
            pid: std::process::id(),
            running: true,
            budget_w: self.settings.budget_w,
            throttle_start_c: c.throttle_start_c,
            hard_c: c.hard_c,
            reason: if reason.is_empty() { "at max".into() } else { reason },
            config_error: self.config_error.clone(),
            gpus,
        };
        if enabled(&c.state_file)
            && let Err(e) = write_atomic(&c.state_file, &serde_json::to_vec_pretty(&state)?)
        {
            eprintln!("gpu-powerd: writing the state file failed: {e:#}");
        }
        Ok(self.last.insert(state))
    }

    pub fn save_models(&self) -> Result<()> {
        let path = &self.settings.config.model_file;
        if !enabled(path) {
            return Ok(());
        }
        let file = ModelFile {
            version: 1,
            uuids: self.backend.devices().iter().map(|d| d.uuid.clone()).collect(),
            models: self.models.clone(),
        };
        write_atomic(path, &serde_json::to_vec_pretty(&file)?)
    }

    /// Put every GPU at its fallback limit; returns the GPUs that failed.
    pub fn restore(&mut self) -> Vec<String> {
        self.restored = true;
        let failures = restore_all(&mut self.backend, &self.settings);
        if let Err(e) = self.save_models() {
            eprintln!("gpu-powerd: saving the model failed: {e:#}");
        }
        let c = &self.settings.config;
        if enabled(&c.state_file) {
            let mut state = self.last.clone();
            if let Some(state) = state.as_mut() {
                state.running = false;
                state.updated = now_unix();
                state.reason = format!(
                    "stopped: fallback {}",
                    self.settings.gpus.iter().map(|g| format!("{:.0} W", g.fallback_w)).collect::<Vec<_>>().join(", ")
                );
                for (gpu, limits) in state.gpus.iter_mut().zip(&self.settings.gpus) {
                    gpu.limit_w = Some(limits.fallback_w.round() as u32);
                }
                let _ = serde_json::to_vec_pretty(state).map(|b| write_atomic(&c.state_file, &b));
            }
        }
        failures
    }
}

impl<B: Backend> Drop for Daemon<B> {
    fn drop(&mut self) {
        // Every exit (return, error, panic unwind) leaves the GPUs at their fallback.
        if !self.restored {
            let failures = self.restore();
            if !failures.is_empty() {
                eprintln!("gpu-powerd: incomplete restore ({}); ExecStopPost retries", failures.join("; "));
            }
        }
    }
}

/// Set every GPU to its fallback limit; returns the failures.
pub fn restore_all<B: Backend + ?Sized>(backend: &mut B, settings: &Settings) -> Vec<String> {
    let mut failures = Vec::new();
    for (i, limits) in settings.gpus.iter().enumerate() {
        let watts = limits.fallback_w.round() as u32;
        match backend.set_limit_w(i, watts) {
            Ok(()) => eprintln!("gpu-powerd: GPU {i} restored to fallback {watts} W"),
            Err(e) => failures.push(format!("GPU {i}: {e}")),
        }
    }
    failures
}
