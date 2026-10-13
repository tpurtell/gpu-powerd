//! The control law: pure, with no I/O, driven once per period.
//!
//! Each GPU's limit is `min(budget share, thermal limit, max_w)`, floored at `base_w`.
//! The thermal limit comes from one of two controllers, with safety bounds no
//! controller can override:
//!
//! - **The learned model, once confident** (and `use_model`): the highest power whose
//!   predicted temperature (steady state, and the trajectory over `model_horizon_s`)
//!   stays at or below `target_c`, given the other cards' current power and the
//!   cross-coupling. A cool card is never capped early; a hot one is held just under
//!   target. A measured-error correction catches a wrong or stale model before
//!   `hard_c`: an observer scales the model's heating by a gain fitted to the measured
//!   temperature (so the plan is made with corrected heating), and a proportional cut
//!   acts while the card is above target. With `model_may_raise = false` the model may only lower
//!   the PI controller's limit.
//! - **Before the model is confident: a PI controller** on smoothed temperature with
//!   setpoint `target_c`. It lowers the limit while above target and releases it while
//!   below; its integral bleeds away below `throttle_start_c`.
//!
//! Below `throttle_start_c` no thermal limit applies at all (the cap goes to `max_w`
//! at the raise rate). At or above `hard_c` (measured or smoothed) a card drops to
//! `base_w`; on a failed or implausible reading to `sensor_fail_w`. Cuts apply at once;
//! the thermal limit rises at most `raise_w_per_s` and a budget share at most
//! `budget_raise_w_per_s`.
//!
//! Budget: every GPU first gets `base_w`; the rest is water-filled (equal levels,
//! max-min fair) up to each GPU's demand, then any spare up to each GPU's thermal
//! limit. With `idle_takes_spare` (the default) an idle GPU holds base plus a share of
//! whatever the busy ones do not want, so it can start work without waiting; the
//! shares rebalance once it draws. Without it an idle GPU stays at base. Demand comes
//! from load: an idle GPU wants base, a power-limited GPU (drawing at least
//! `pinned_fraction` of its enforced limit, or reporting a power-cap clock event while
//! busy) wants its thermal limit, and any other busy GPU wants its draw plus
//! `demand_headroom_w`. Demand never exceeds the thermal limit, so a GPU capped for
//! heat releases its budget to the others.
use crate::backend::Reading;
use crate::config::Settings;
use crate::model::ThermalModel;
use serde::Serialize;

/// What set a GPU's limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bound {
    /// A failed or implausible reading: sensor_fail_w.
    Sensor,
    /// At or above hard_c: base.
    Hard,
    /// The PI controller toward target_c (no confident model).
    Pi,
    /// The learned model's predicted temperature.
    Model,
    /// The model, trimmed by the measured-error correction.
    Correction,
    /// The thermal limit is rising back at the slow rate.
    Recovering,
    /// The GPU's share of the system budget.
    Budget,
    /// Unconstrained: max_w.
    Max,
}

#[derive(Clone, Debug)]
struct GpuState {
    temp_smoothed: Option<f64>,
    power_smoothed: f64,
    thermal_cap: Option<f64>,
    share: Option<f64>,
    /// PI integral (watts of reduction below max_w).
    pi_integral: f64,
    /// The observer's heating gain: measured over model-predicted heating.
    gain: f64,
    /// The model observer: the temperature the model expects now, given measured powers.
    predicted_c: Option<f64>,
}

impl Default for GpuState {
    fn default() -> Self {
        Self {
            temp_smoothed: None,
            power_smoothed: 0.0,
            thermal_cap: None,
            share: None,
            pi_integral: 0.0,
            gain: 1.0,
            predicted_c: None,
        }
    }
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
    /// The PI controller's limit (always computed, so it is ready as a fallback).
    pub pi_w: f64,
    /// The model's planned power, when a confident model acts.
    pub model_w: Option<f64>,
    /// Watts the measured-error correction removes from the model's plan.
    pub correction_w: f64,
    /// The temperature the model expects now (its observer), when it acts.
    pub predicted_c: Option<f64>,
    /// The observer's heating gain (measured over modelled heating; 1 = the model is right).
    pub model_gain: f64,
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

/// The observer's heating gain stays within 1/MAX_GAIN..MAX_GAIN.
const MAX_GAIN: f64 = 4.0;
/// The gain adapts against at least this much modelled heating (°C above ambient).
const MIN_RISE_C: f64 = 5.0;

fn alpha(dt: f64, tau: f64) -> f64 {
    if tau <= 0.0 { 1.0 } else { 1.0 - (-dt / tau).exp() }
}

impl Controller {
    pub fn new(n: usize) -> Self {
        Self { gpus: vec![GpuState::default(); n] }
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
                // Restart smoothing (and the PI state) from the next good reading.
                Err(_) => {
                    state.temp_smoothed = None;
                    state.predicted_c = None;
                }
            }
        }
        // Planning uses the other cards' current (instant) draw, which is what they will
        // heat this card with next.
        let powers: Vec<f64> = checked
            .iter()
            .zip(&self.gpus)
            .map(|(r, g)| r.as_ref().map(|r| r.power_w).unwrap_or(g.power_smoothed))
            .collect();

        // Thermal limits.
        struct Thermal {
            target: f64,
            pi: f64,
            model: Option<f64>,
            correction: f64,
            predicted: Option<f64>,
            hard: bool,
            cool: bool,
        }
        let mut thermal = Vec::with_capacity(n);
        for i in 0..n {
            let card = &settings.gpus[i];
            let (base, max) = (card.base_w, card.max_w);
            let state = &mut self.gpus[i];
            let t = match &checked[i] {
                Ok(r) => {
                    let smoothed = state.temp_smoothed.unwrap_or(r.temperature_c);
                    // Control on the hotter of measured and smoothed: smoothing must not
                    // hide a rise.
                    let temp = smoothed.max(r.temperature_c);
                    let hard = r.temperature_c >= card.hard_c || smoothed >= card.hard_c;
                    let cool = temp < card.throttle_start_c;

                    // PI toward target_c: the integral (watts below max) grows while above
                    // target, shrinks while below, and bleeds away below throttle_start_c.
                    let error = temp - card.target_c;
                    if cool {
                        state.pi_integral *= (-dt / 10.0).exp();
                    } else {
                        state.pi_integral += card.pi_ki_w_per_c_s * error * dt;
                    }
                    state.pi_integral = state.pi_integral.clamp(0.0, max - base);
                    let pi = if cool {
                        max
                    } else {
                        (max - state.pi_integral - card.pi_kp_w_per_c * error.max(0.0)).clamp(base, max)
                    };

                    // The model, with the measured-error correction. An observer runs the
                    // model forward from the measured powers with the heating (every R
                    // term) scaled by a gain `g`, and adapts `g` toward what the measured
                    // temperature implies: a model whose R is off by 2x is corrected by
                    // g = 2 (or 0.5) within a few time constants, and its plan, made with
                    // that gain, then holds target_c exactly. The gain adapts at
                    // `correction_rate_per_s` (fraction of the error per second), four times
                    // faster when the card runs hotter than predicted (the unsafe
                    // direction). A proportional term (`correction_kp_w_per_c`) trims the
                    // plan at once while the card is above target.
                    let (model, correction, predicted) = match models[i].filter(|_| card.use_model) {
                        Some(m) => {
                            let others = &powers;
                            let gain = state.gain;
                            let observer = match state.predicted_c {
                                Some(previous) => m.predict_scaled_c(previous, r.power_w, others, dt, gain),
                                None => smoothed,
                            };
                            let error = smoothed - observer;
                            // The heating the observer attributes now; the gain moves by the
                            // error relative to it, so its step size is in °C, not in units of g.
                            let rise = (observer - m.t_amb_c()).max(MIN_RISE_C);
                            let rate = card.correction_rate_per_s * if error > 0.0 { 4.0 } else { 1.0 };
                            let new_gain =
                                (gain * (1.0 + (rate * dt).min(1.0) * error / rise)).clamp(1.0 / MAX_GAIN, MAX_GAIN);
                            // Re-anchor the observer to the measurement as the gain absorbs it.
                            let observer = observer + (rate * dt).min(1.0) * error;
                            state.gain = new_gain;
                            state.predicted_c = Some(observer);
                            let plan =
                                m.planned_scaled_power_w(card.target_c, temp, others, c.model_horizon_s, new_gain);
                            let over = smoothed - card.target_c;
                            let correction = card.correction_kp_w_per_c * over.max(0.0);
                            (Some(plan), correction, Some(observer))
                        }
                        None => {
                            state.gain = 1.0;
                            state.predicted_c = None;
                            (None, 0.0, None)
                        }
                    };
                    let target = if hard {
                        base
                    } else if cool && model.is_none() {
                        // Below throttle_start_c the PI controller applies no limit at all.
                        // (A confident model may still cap a card it predicts will pass
                        // target_c: that is the preventive limit.)
                        max
                    } else {
                        match model {
                            Some(plan) => {
                                let corrected = plan - correction;
                                if card.model_may_raise { corrected } else { corrected.min(pi) }
                            }
                            None => pi,
                        }
                        .clamp(base, max)
                    };
                    Thermal { target, pi, model, correction, predicted, hard, cool }
                }
                Err(_) => {
                    state.pi_integral = 0.0;
                    let fail = card.sensor_fail_w;
                    Thermal {
                        target: fail,
                        pi: fail,
                        model: None,
                        correction: 0.0,
                        predicted: None,
                        hard: false,
                        cool: false,
                    }
                }
            };
            let start = match (&checked[i], state.thermal_cap) {
                (_, Some(cap)) => cap,
                (Ok(r), None) => r.limit_w.clamp(base, max),
                (Err(_), None) => card.sensor_fail_w,
            };
            let cap = if t.target <= start { t.target } else { t.target.min(start + c.raise_w_per_s * dt) };
            state.thermal_cap = Some(if checked[i].is_ok() { cap.clamp(base, max) } else { card.sensor_fail_w });
            thermal.push(t);
        }

        // Demand and the budget.
        let mut demand = vec![0.0; n];
        let mut pinned = vec![false; n];
        let mut idle = vec![false; n];
        let caps: Vec<f64> = self.gpus.iter().map(|g| g.thermal_cap.unwrap_or(0.0)).collect();
        let bases: Vec<f64> = settings.gpus.iter().map(|g| g.base_w).collect();
        // A failed card is held at sensor_fail_w: reserve it from the budget.
        let floors: Vec<f64> = (0..n)
            .map(|i| if checked[i].is_ok() { bases[i] } else { settings.gpus[i].sensor_fail_w.min(bases[i]) })
            .collect();
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
                Err(_) => floors[i],
            };
        }
        let ok: Vec<bool> = checked.iter().map(Result::is_ok).collect();
        let ceilings_demand: Vec<f64> = (0..n).map(|i| if ok[i] { demand[i] } else { floors[i] }).collect();
        // Spare budget goes up to each thermal limit; an idle GPU takes a share of it only
        // with idle_takes_spare (otherwise it stays at its demand, base).
        let ceilings_cap: Vec<f64> = (0..n)
            .map(|i| match ok[i] {
                true if idle[i] && !c.idle_takes_spare => demand[i],
                true => caps[i],
                false => floors[i],
            })
            .collect();
        let first = water_fill(&floors, &ceilings_demand, settings.budget_w);
        let fill = water_fill(&first, &ceilings_cap, settings.budget_w);

        let mut plans = Vec::with_capacity(n);
        for i in 0..n {
            let card = &settings.gpus[i];
            let (base, max) = (card.base_w, card.max_w);
            let state = &mut self.gpus[i];
            let t = &thermal[i];
            let smoothed = state.temp_smoothed;
            if let Err(error) = &checked[i] {
                // Sensor failure: sensor_fail_w regardless of budget or rates.
                state.share = None;
                plans.push(GpuPlan {
                    limit_w: card.sensor_fail_w,
                    target_w: card.sensor_fail_w,
                    bound: Bound::Sensor,
                    reason: format!("sensor failure: {error}"),
                    sensor_ok: false,
                    sensor_error: Some(error.clone()),
                    temperature_smoothed_c: None,
                    power_smoothed_w: state.power_smoothed,
                    pi_w: t.pi,
                    model_w: None,
                    correction_w: 0.0,
                    predicted_c: None,
                    model_gain: 1.0,
                    thermal_cap_w: card.sensor_fail_w,
                    demand_w: demand[i],
                    share_w: card.sensor_fail_w,
                    pinned: false,
                    idle: false,
                });
                continue;
            }
            let previous = match (&checked[i], state.share) {
                (_, Some(share)) => share,
                (Ok(r), None) => r.limit_w.clamp(base, max),
                (Err(_), None) => base,
            };
            let share = if fill[i] <= previous { fill[i] } else { fill[i].min(previous + c.budget_raise_w_per_s * dt) };
            state.share = Some(share);
            let cap = caps[i];
            let limit = share.min(cap).min(max).max(base);
            let target = fill[i].min(t.target).min(max).max(base);
            let near = |a: f64, b: f64| (a - b).abs() < 0.5;
            let r = checked[i].as_ref().ok();
            let (bound, reason) = if t.hard {
                let r = r.map(|r| r.temperature_c).unwrap_or_default();
                (
                    Bound::Hard,
                    format!("hard: {r:.0} C (smoothed {:.1} C) >= {:.0} C", smoothed.unwrap_or(0.0), card.hard_c),
                )
            } else if near(limit, max) && share >= max - 0.5 && cap >= max - 0.5 {
                (Bound::Max, "max".into())
            } else if share < cap - 0.5 && idle[i] {
                (Bound::Budget, format!("idle: base plus spare budget, {:.0} W of {:.0} W", share, settings.budget_w))
            } else if share < cap - 0.5 {
                (
                    Bound::Budget,
                    format!("budget: share {:.0} W of {:.0} W (demand {:.0} W)", share, settings.budget_w, demand[i]),
                )
            } else if cap < t.target - 0.5 {
                (Bound::Recovering, format!("recovering: +{:.0} W/s toward {:.0} W", c.raise_w_per_s, t.target))
            } else if let Some(plan) = t.model {
                if t.correction > 0.5 {
                    (
                        Bound::Correction,
                        format!(
                            "model {:.0} W less {:.0} W measured-error correction ({:.1} C, target {:.0} C)",
                            plan,
                            t.correction,
                            smoothed.unwrap_or(0.0),
                            card.target_c
                        ),
                    )
                } else {
                    (
                        Bound::Model,
                        format!(
                            "model: {:.0} W keeps the predicted temperature <= {:.0} C (gain {:.2})",
                            plan, card.target_c, state.gain
                        ),
                    )
                }
            } else {
                (
                    Bound::Pi,
                    format!(
                        "pi: {:.1} C toward target {:.0} C{}",
                        smoothed.unwrap_or(0.0),
                        card.target_c,
                        if t.cool { " (releasing)" } else { "" }
                    ),
                )
            };
            plans.push(GpuPlan {
                limit_w: limit,
                target_w: target,
                bound,
                reason,
                sensor_ok: true,
                sensor_error: None,
                temperature_smoothed_c: smoothed,
                power_smoothed_w: state.power_smoothed,
                pi_w: t.pi,
                model_w: t.model,
                correction_w: t.correction,
                predicted_c: t.predicted,
                model_gain: state.gain,
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
    fn water_fill_and_quantise() {
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
        let good = Reading { fan_pct: Some(40.0), ..Reading::basic(50.0, 300.0, 90.0, 600.0) };
        assert!(plausible(&good).is_ok());
        assert!(plausible(&Reading { temperature_c: 0.0, ..good }).is_err());
        assert!(plausible(&Reading { temperature_c: 255.0, ..good }).is_err());
        assert!(plausible(&Reading { power_w: f64::NAN, ..good }).is_err());
        assert!(plausible(&Reading { utilization_pct: 101.0, ..good }).is_err());
        assert!(plausible(&Reading { fan_pct: Some(-1.0), ..good }).is_err());
    }
}
