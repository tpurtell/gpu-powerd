//! The control law: pure, with no I/O, driven once per period.
//!
//! Per GPU the final limit is `min(budget share, thermal cap, max_w)`, floored at
//! `base_w`, where the thermal cap is `min(ramp(T_smoothed), model_safe_power)`.
//! Cuts apply at once; the thermal cap rises at most `raise_w_per_s` and a budget
//! share at most `budget_raise_w_per_s`. At or above `hard_c` (measured or smoothed)
//! and on any failed or implausible reading a GPU goes straight to `base_w`.
//!
//! Budget: every GPU first gets `base_w`; the rest is water-filled (equal levels,
//! max-min fair) up to each GPU's demand, then any spare up to each GPU's thermal
//! cap. With `idle_takes_spare` (the default) an idle GPU holds base plus a share of
//! whatever the busy ones do not want, so it can start work without waiting; the
//! shares rebalance once it draws. Without it an idle GPU stays at base. Demand comes from load: an idle GPU wants base, a power-limited GPU
//! (drawing at least `pinned_fraction` of its enforced limit, or reporting a power
//! cap clock event while busy) wants its thermal cap, and any other busy GPU wants
//! its draw plus `demand_headroom_w`. Demand never exceeds the thermal cap, so a
//! GPU capped for heat releases its budget to the others.
use crate::backend::Reading;
use crate::config::Settings;
use crate::model::ThermalModel;
use serde::Serialize;

/// What set a GPU's limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bound {
    /// A failed or implausible reading: base.
    Sensor,
    /// At or above hard_c: base.
    Hard,
    /// The temperature ramp.
    Ramp,
    /// The thermal model's predicted steady state.
    Model,
    /// The thermal cap is rising back at the slow rate.
    Recovering,
    /// The GPU's share of the system budget.
    Budget,
    /// Unconstrained: max_w.
    Max,
}

#[derive(Clone, Debug, Default)]
struct GpuState {
    temp_smoothed: Option<f64>,
    power_smoothed: f64,
    thermal_cap: Option<f64>,
    share: Option<f64>,
}

/// One GPU's decision for this period.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct GpuPlan {
    /// The limit to apply now (watts, before quantisation).
    pub limit_w: f64,
    /// Where the limit is heading without rate limits.
    pub target_w: f64,
    pub bound: Bound,
    pub reason: String,
    pub sensor_ok: bool,
    pub sensor_error: Option<String>,
    pub temperature_smoothed_c: Option<f64>,
    pub power_smoothed_w: f64,
    pub ramp_w: f64,
    pub model_w: Option<f64>,
    pub thermal_cap_w: f64,
    pub demand_w: f64,
    pub share_w: f64,
    pub pinned: bool,
    pub idle: bool,
}

/// The controller for every GPU.
#[derive(Clone, Debug, Default)]
pub struct Controller {
    gpus: Vec<GpuState>,
}

/// Reject readings no working sensor produces.
pub fn plausible(reading: &Reading) -> Result<(), String> {
    let Reading { temperature_c: t, power_w: p, utilization_pct: u, limit_w: l, fan_pct, .. } = *reading;
    if !(t.is_finite() && t > 0.0 && t < 125.0) {
        return Err(format!("implausible temperature {t} C"));
    }
    if !(p.is_finite() && (0.0..=2000.0).contains(&p)) {
        return Err(format!("implausible power {p} W"));
    }
    if !(u.is_finite() && (0.0..=100.0).contains(&u)) {
        return Err(format!("implausible utilization {u}%"));
    }
    if !(l.is_finite() && l > 0.0 && l <= 2000.0) {
        return Err(format!("implausible power limit {l} W"));
    }
    if let Some(f) = fan_pct
        && !(f.is_finite() && (0.0..=100.0).contains(&f))
    {
        return Err(format!("implausible fan speed {f}%"));
    }
    Ok(())
}

/// The ramp: `max_w` at or below `start`, `base_w` at or above `hard`, linear between.
pub fn ramp(temp_c: f64, start: f64, hard: f64, base_w: f64, max_w: f64) -> f64 {
    if temp_c <= start {
        max_w
    } else if temp_c >= hard {
        base_w
    } else {
        max_w - (max_w - base_w) * (temp_c - start) / (hard - start)
    }
}

/// Max-min fair fill: `alloc_i = clamp(L, floor_i, ceil_i)` with the largest level `L`
/// whose total fits `budget` (or every ceiling, when that fits).
pub fn water_fill(floors: &[f64], ceilings: &[f64], budget: f64) -> Vec<f64> {
    let at =
        |level: f64| -> Vec<f64> { floors.iter().zip(ceilings).map(|(&f, &c)| level.max(f).min(c.max(f))).collect() };
    let total = |v: &[f64]| v.iter().sum::<f64>();
    let top = ceilings.iter().chain(floors).cloned().fold(0.0, f64::max);
    if total(&at(top)) <= budget + 1e-6 {
        return at(top);
    }
    let (mut low, mut high) = (0.0, top);
    for _ in 0..64 {
        let mid = 0.5 * (low + high);
        if total(&at(mid)) <= budget { low = mid } else { high = mid }
    }
    at(low)
}

fn alpha(dt: f64, tau: f64) -> f64 {
    if tau <= 0.0 { 1.0 } else { 1.0 - (-dt / tau).exp() }
}

impl Controller {
    pub fn new(n: usize) -> Self {
        Self { gpus: vec![GpuState::default(); n] }
    }

    /// Forget rate history (thermal caps and shares restart from the enforced limits).
    pub fn reset(&mut self) {
        let n = self.gpus.len();
        *self = Self::new(n);
    }

    /// One period. `readings[i]` is GPU i's reading or why it failed; `models[i]` is a
    /// model allowed to act (confident, enabled), if any.
    pub fn step(
        &mut self,
        settings: &Settings,
        readings: &[Result<Reading, String>],
        models: &[Option<&ThermalModel>],
        dt: f64,
    ) -> Vec<GpuPlan> {
        let c = &settings.config;
        let n = self.gpus.len();
        assert_eq!(readings.len(), n);
        assert_eq!(settings.gpus.len(), n);
        let checked: Vec<Result<Reading, String>> =
            readings.iter().map(|r| r.clone().and_then(|r| plausible(&r).map(|()| r))).collect();
        // Smoothing.
        let a = alpha(dt, c.smoothing_s);
        for (state, reading) in self.gpus.iter_mut().zip(&checked) {
            match reading {
                Ok(r) => {
                    state.temp_smoothed = Some(match state.temp_smoothed {
                        Some(t) => t + a * (r.temperature_c - t),
                        None => r.temperature_c,
                    });
                    state.power_smoothed = if state.thermal_cap.is_none() {
                        r.power_w
                    } else {
                        state.power_smoothed + a * (r.power_w - state.power_smoothed)
                    };
                }
                // Restart smoothing from the next good reading.
                Err(_) => state.temp_smoothed = None,
            }
        }
        let powers: Vec<f64> = self.gpus.iter().map(|g| g.power_smoothed).collect();
        let target_c = c.throttle_start_c - c.model_margin_c;

        // Thermal caps.
        struct Thermal {
            target: f64,
            ramp: f64,
            model: Option<f64>,
            hard: bool,
        }
        let mut thermal = Vec::with_capacity(n);
        for i in 0..n {
            let limits = &settings.gpus[i];
            let (base, max) = (limits.base_w, limits.max_w);
            let t = match &checked[i] {
                Ok(r) => {
                    let smoothed = self.gpus[i].temp_smoothed.unwrap_or(r.temperature_c);
                    let ramp_w = ramp(smoothed, c.throttle_start_c, c.hard_c, base, max);
                    let model_w = models[i].map(|m| m.safe_power_w(target_c, &powers));
                    let hard = r.temperature_c >= c.hard_c || smoothed >= c.hard_c;
                    let target =
                        if hard { base } else { ramp_w.min(model_w.unwrap_or(f64::INFINITY)).min(max).max(base) };
                    Thermal { target, ramp: ramp_w, model: model_w, hard }
                }
                Err(_) => Thermal { target: base, ramp: base, model: None, hard: false },
            };
            let state = &mut self.gpus[i];
            let start = match (&checked[i], state.thermal_cap) {
                (_, Some(cap)) => cap,
                (Ok(r), None) => r.limit_w.clamp(base, max),
                (Err(_), None) => base,
            };
            let cap = if t.target <= start { t.target } else { t.target.min(start + c.raise_w_per_s * dt) };
            state.thermal_cap = Some(cap.clamp(base, max));
            thermal.push(t);
        }

        // Demand and the budget.
        let mut demand = vec![0.0; n];
        let mut pinned = vec![false; n];
        let mut idle = vec![false; n];
        let caps: Vec<f64> = self.gpus.iter().map(|g| g.thermal_cap.unwrap_or(0.0)).collect();
        let bases: Vec<f64> = settings.gpus.iter().map(|g| g.base_w).collect();
        for i in 0..n {
            let base = bases[i];
            demand[i] = match &checked[i] {
                Ok(r) => {
                    let busy = r.utilization_pct >= c.idle_util_pct;
                    // The larger of the instant and smoothed draw: smoothing lags a limit
                    // that is rising, and a card climbing to its share is still pinned.
                    let draw = r.power_w.max(self.gpus[i].power_smoothed);
                    pinned[i] = draw >= c.pinned_fraction * r.limit_w || (busy && r.power_capped == Some(true));
                    idle[i] = !busy && !pinned[i];
                    if pinned[i] {
                        caps[i]
                    } else if idle[i] {
                        base
                    } else {
                        (draw + c.demand_headroom_w).clamp(base, caps[i])
                    }
                }
                Err(_) => base,
            };
        }
        let ok: Vec<bool> = checked.iter().map(Result::is_ok).collect();
        let ceilings_demand: Vec<f64> = (0..n).map(|i| if ok[i] { demand[i] } else { bases[i] }).collect();
        // Spare budget goes up to each thermal cap; an idle GPU takes a share of it only
        // with idle_takes_spare (otherwise it stays at its demand, base).
        let ceilings_cap: Vec<f64> = (0..n)
            .map(|i| match ok[i] {
                true if idle[i] && !c.idle_takes_spare => demand[i],
                true => caps[i],
                false => bases[i],
            })
            .collect();
        let first = water_fill(&bases, &ceilings_demand, settings.budget_w);
        let fill = water_fill(&first, &ceilings_cap, settings.budget_w);

        let mut plans = Vec::with_capacity(n);
        for i in 0..n {
            let limits = &settings.gpus[i];
            let (base, max) = (limits.base_w, limits.max_w);
            let state = &mut self.gpus[i];
            let previous = match (&checked[i], state.share) {
                (_, Some(share)) => share,
                (Ok(r), None) => r.limit_w.clamp(base, max),
                (Err(_), None) => base,
            };
            let share = if fill[i] <= previous { fill[i] } else { fill[i].min(previous + c.budget_raise_w_per_s * dt) };
            state.share = Some(share);
            let cap = caps[i];
            let limit = share.min(cap).min(max).max(base);
            let t = &thermal[i];
            let target = fill[i].min(t.target).min(max).max(base);
            let smoothed = state.temp_smoothed;
            let near = |a: f64, b: f64| (a - b).abs() < 0.5;
            let (bound, reason) = match &checked[i] {
                Err(error) => (Bound::Sensor, format!("sensor failure: {error}")),
                Ok(r) if t.hard => (
                    Bound::Hard,
                    format!(
                        "hard: {:.0} C (smoothed {:.1} C) >= {:.0} C",
                        r.temperature_c,
                        smoothed.unwrap_or(0.0),
                        c.hard_c
                    ),
                ),
                Ok(_) if near(limit, max) && share >= max - 0.5 && cap >= max - 0.5 => (Bound::Max, "max".into()),
                Ok(_) if share < cap - 0.5 && idle[i] => (
                    Bound::Budget,
                    format!("idle: base plus spare budget, {:.0} W of {:.0} W", share, settings.budget_w),
                ),
                Ok(_) if share < cap - 0.5 => (
                    Bound::Budget,
                    format!("budget: share {:.0} W of {:.0} W (demand {:.0} W)", share, settings.budget_w, demand[i]),
                ),
                Ok(_) if cap < t.target - 0.5 => {
                    (Bound::Recovering, format!("recovering: +{:.0} W/s toward {:.0} W", c.raise_w_per_s, t.target))
                }
                Ok(_) if t.model.is_some_and(|m| m < t.ramp) => (
                    Bound::Model,
                    format!(
                        "model: steady state at {:.0} W predicted {:.1} C <= {:.1} C",
                        t.target,
                        models[i].map(|m| m.steady_state_c(t.target, &powers)).unwrap_or(f64::NAN),
                        target_c
                    ),
                ),
                Ok(_) => (
                    Bound::Ramp,
                    format!(
                        "ramp: {:.1} C between {:.0} and {:.0} C",
                        smoothed.unwrap_or(0.0),
                        c.throttle_start_c,
                        c.hard_c
                    ),
                ),
            };
            plans.push(GpuPlan {
                limit_w: limit,
                target_w: target,
                bound,
                reason,
                sensor_ok: checked[i].is_ok(),
                sensor_error: checked[i].as_ref().err().cloned(),
                temperature_smoothed_c: smoothed,
                power_smoothed_w: state.power_smoothed,
                ramp_w: t.ramp,
                model_w: t.model,
                thermal_cap_w: cap,
                demand_w: demand[i],
                share_w: share,
                pinned: pinned[i],
                idle: idle[i],
            });
        }
        plans
    }
}

/// The limit to write: whole multiples of `step` (rounded down), within `[base, max]`.
pub fn quantise(limit_w: f64, step_w: f64, base_w: f64, max_w: f64) -> u32 {
    let stepped = (limit_w / step_w + 1e-6).floor() * step_w;
    stepped.max(base_w.ceil()).min(max_w.floor()) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_and_water_fill() {
        assert_eq!(ramp(70.0, 75.0, 85.0, 150.0, 600.0), 600.0);
        assert_eq!(ramp(80.0, 75.0, 85.0, 150.0, 600.0), 375.0);
        assert_eq!(ramp(90.0, 75.0, 85.0, 150.0, 600.0), 150.0);
        let near = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
        assert!(near(&water_fill(&[150.0, 150.0], &[600.0, 600.0], 900.0), &[450.0, 450.0]));
        assert!(near(&water_fill(&[150.0, 150.0], &[600.0, 300.0], 900.0), &[600.0, 300.0]));
        assert!(near(&water_fill(&[150.0, 150.0], &[600.0, 150.0], 900.0), &[600.0, 150.0]));
        assert!(near(&water_fill(&[150.0, 150.0], &[600.0, 600.0], 1200.0), &[600.0, 600.0]));
        assert!(near(&water_fill(&[150.0, 150.0], &[600.0, 600.0], 300.0), &[150.0, 150.0]));
        assert_eq!(quantise(447.9, 5.0, 150.0, 600.0), 445);
        assert_eq!(quantise(100.0, 5.0, 150.0, 600.0), 150);
        assert_eq!(quantise(700.0, 5.0, 150.0, 600.0), 600);
    }

    #[test]
    fn rejects_implausible_readings() {
        let good = Reading {
            temperature_c: 50.0,
            power_w: 300.0,
            utilization_pct: 90.0,
            limit_w: 600.0,
            fan_pct: Some(40.0),
            power_capped: None,
        };
        assert!(plausible(&good).is_ok());
        assert!(plausible(&Reading { temperature_c: 0.0, ..good }).is_err());
        assert!(plausible(&Reading { temperature_c: 255.0, ..good }).is_err());
        assert!(plausible(&Reading { power_w: f64::NAN, ..good }).is_err());
        assert!(plausible(&Reading { utilization_pct: 101.0, ..good }).is_err());
        assert!(plausible(&Reading { fan_pct: Some(-1.0), ..good }).is_err());
    }
}
