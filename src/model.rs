//! A first-order thermal model per GPU, fitted online by recursive least squares.
//!
//! Physics: `τ_i · dT_i/dt = T_amb + Σ_j R_ij · P_j − T_i`, so in steady state
//! `T_i → T_amb + R_ii·P_i + Σ_{j≠i} R_ij·P_j` with time constant `τ_i`. Sampled every
//! `Δ` seconds (one window of 1 Hz readings) with the power held over the window,
//! that is exactly linear in the parameters:
//!
//! `T[k+1] = a·T[k] + (1−a)·T_amb + (1−a)·Σ_j R_ij·P̄_j[k+1]`,  `a = exp(−Δ/τ)`.
//!
//! RLS fits `[a, c, g_0 … g_{n−1}]` on centred, scaled regressors (`(T−50)/10`, `1`,
//! `P_j/100`) and maps them back to `R`, `τ` and `T_amb`. Forgetting is applied only
//! while the inputs move, so long idle stretches do not wind the covariance up and
//! erase what was learned under load. Fan speed is recorded with every sample (the
//! fit's mean fan is reported) but not controlled; its effect is folded into `R`.
use serde::{Deserialize, Serialize};

const T_CENTRE: f64 = 50.0;
const T_SCALE: f64 = 10.0;
const P_SCALE: f64 = 100.0;
/// Prior: a moderate card in a ventilated case. Weak, so data dominates within minutes.
const PRIOR_TAU_S: f64 = 120.0;
const PRIOR_T_AMB: f64 = 30.0;
const PRIOR_R_SELF: f64 = 0.08;
const PRIOR_R_CROSS: f64 = 0.01;
/// Plausibility bounds outside which the model never acts.
const TAU_RANGE: (f64, f64) = (5.0, 3600.0);
const R_SELF_RANGE: (f64, f64) = (0.005, 1.0);
const T_AMB_RANGE: (f64, f64) = (0.0, 70.0);
/// Exponential averaging factor for the residual (≈ 50 windows).
const RESIDUAL_ALPHA: f64 = 0.02;
/// Inputs "move" when a window's mean power changes this much (W) or temperature this much (°C).
const MOVE_POWER_W: f64 = 5.0;
const MOVE_TEMP_C: f64 = 0.5;
/// Covariance trace ceiling (guards against windup if forgetting ever runs unexcited).
const MAX_TRACE: f64 = 100.0;

/// Fit settings shared by every GPU's model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitSettings {
    pub window_s: f64,
    pub forgetting: f64,
    pub min_samples: u64,
    pub min_power_std_w: f64,
    pub max_residual_c: f64,
}

/// One GPU's fitted model (persisted).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ThermalModel {
    /// Index of this GPU among the regressors' powers.
    pub gpu: usize,
    pub n_gpus: usize,
    pub window_s: f64,
    /// `[a, c, g_0 … g_{n−1}]` in scaled units.
    pub theta: Vec<f64>,
    /// Row-major covariance.
    pub covariance: Vec<f64>,
    /// Windows fitted.
    pub samples: u64,
    /// Exponentially weighted statistics of this GPU's window power.
    pub power_weight: f64,
    pub power_mean_w: f64,
    pub power_m2: f64,
    /// Exponentially weighted mean square one-step prediction error (°C²).
    pub residual_ms: f64,
    /// Mean fan speed over the fit (percent), when the board reports fans.
    pub fan_mean_pct: Option<f64>,
    /// The previous window's powers (decides whether the inputs moved).
    #[serde(default)]
    pub last_powers: Vec<f64>,
}

/// A readable summary (R, τ, confidence, samples) for status output and the state file.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelSummary {
    pub r_self_c_per_w: f64,
    /// `R_ij` for every GPU `j` (this GPU's own entry is `r_self`).
    pub r_c_per_w: Vec<f64>,
    pub tau_s: f64,
    pub t_amb_c: f64,
    pub samples: u64,
    pub power_std_w: f64,
    pub residual_c: f64,
    pub fan_mean_pct: Option<f64>,
    /// 0..1: the fraction of the sample and power-variation requirements met.
    pub confidence: f64,
    pub plausible: bool,
    /// The model may lower limits.
    pub confident: bool,
}

impl ThermalModel {
    pub fn new(gpu: usize, n_gpus: usize, window_s: f64) -> Self {
        let m = 2 + n_gpus;
        let a = (-window_s / PRIOR_TAU_S).exp();
        let mut theta = vec![0.0; m];
        theta[0] = a;
        theta[1] = (PRIOR_T_AMB - T_CENTRE) * (1.0 - a) / T_SCALE;
        for j in 0..n_gpus {
            let r = if j == gpu { PRIOR_R_SELF } else { PRIOR_R_CROSS };
            theta[2 + j] = r * P_SCALE * (1.0 - a) / T_SCALE;
        }
        let mut covariance = vec![0.0; m * m];
        let diag = |i: usize| match i {
            0 => 1e-2,
            _ => 1e-1,
        };
        for i in 0..m {
            covariance[i * m + i] = diag(i);
        }
        Self {
            gpu,
            n_gpus,
            window_s,
            theta,
            covariance,
            samples: 0,
            power_weight: 0.0,
            power_mean_w: 0.0,
            power_m2: 0.0,
            residual_ms: 0.0,
            fan_mean_pct: None,
            last_powers: Vec::new(),
        }
    }

    /// A model with the given physical parameters (`r[j]` = `R_gpu,j`), for tests and tools.
    pub fn with_params(gpu: usize, window_s: f64, tau_s: f64, t_amb_c: f64, r: &[f64]) -> Self {
        let mut model = Self::new(gpu, r.len(), window_s);
        let a = (-window_s / tau_s).exp();
        model.theta[0] = a;
        model.theta[1] = (t_amb_c - T_CENTRE) * (1.0 - a) / T_SCALE;
        for (j, r) in r.iter().enumerate() {
            model.theta[2 + j] = r * P_SCALE * (1.0 - a) / T_SCALE;
        }
        model
    }

    fn dim(&self) -> usize {
        2 + self.n_gpus
    }

    /// A persisted model is usable when its shape matches.
    pub fn compatible(&self, gpu: usize, n_gpus: usize, window_s: f64) -> bool {
        self.gpu == gpu
            && self.n_gpus == n_gpus
            && (self.window_s - window_s).abs() < 1e-9
            && self.theta.len() == self.dim()
            && self.covariance.len() == self.dim() * self.dim()
            && self.theta.iter().chain(&self.covariance).all(|v| v.is_finite())
    }

    /// Fit one window: `t_prev` and `t_next` are this GPU's temperature at the ends of
    /// consecutive windows, `powers` every GPU's mean power over the second window.
    pub fn update(&mut self, t_prev: f64, t_next: f64, powers: &[f64], fan_pct: Option<f64>, fit: &FitSettings) {
        debug_assert_eq!(powers.len(), self.n_gpus);
        let m = self.dim();
        let mut phi = vec![0.0; m];
        phi[0] = (t_prev - T_CENTRE) / T_SCALE;
        phi[1] = 1.0;
        for (j, p) in powers.iter().enumerate() {
            phi[2 + j] = p / P_SCALE;
        }
        let y = (t_next - T_CENTRE) / T_SCALE;
        let error = y - dot(&phi, &self.theta);
        let moved = (t_next - t_prev).abs() >= MOVE_TEMP_C
            || self.last_powers.len() != powers.len()
            || self.last_powers.iter().zip(powers).any(|(a, b)| (a - b).abs() >= MOVE_POWER_W);
        self.last_powers = powers.to_vec();
        let lambda = if moved { fit.forgetting } else { 1.0 };

        // RLS: k = Pφ / (λ + φᵀPφ); θ += k·e; P = (P − k·φᵀP) / λ.
        let p_phi: Vec<f64> = (0..m).map(|i| (0..m).map(|j| self.covariance[i * m + j] * phi[j]).sum()).collect();
        let denominator = lambda + dot(&phi, &p_phi);
        if !(denominator.is_finite() && denominator > 0.0) {
            return;
        }
        let gain: Vec<f64> = p_phi.iter().map(|v| v / denominator).collect();
        for (theta, gain) in self.theta.iter_mut().zip(&gain) {
            *theta += gain * error;
        }
        let mut next = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                next[i * m + j] = (self.covariance[i * m + j] - gain[i] * p_phi[j]) / lambda;
            }
        }
        // Keep it symmetric and bounded.
        for i in 0..m {
            for j in 0..i {
                let v = 0.5 * (next[i * m + j] + next[j * m + i]);
                next[i * m + j] = v;
                next[j * m + i] = v;
            }
        }
        let trace: f64 = (0..m).map(|i| next[i * m + i]).sum();
        if trace > MAX_TRACE {
            next.iter_mut().for_each(|v| *v *= MAX_TRACE / trace);
        }
        if next.iter().chain(&self.theta).all(|v| v.is_finite()) {
            self.covariance = next;
        } else {
            // Numerical trouble: start over rather than act on garbage.
            *self = Self::new(self.gpu, self.n_gpus, self.window_s);
            return;
        }

        self.samples += 1;
        let error_c = error * T_SCALE;
        self.residual_ms = if self.samples == 1 {
            error_c * error_c
        } else {
            self.residual_ms + RESIDUAL_ALPHA * (error_c * error_c - self.residual_ms)
        };
        // Exponentially weighted power variance (same forgetting as the fit).
        let p = powers[self.gpu];
        self.power_weight = lambda * self.power_weight + 1.0;
        let delta = p - self.power_mean_w;
        self.power_mean_w += delta / self.power_weight;
        self.power_m2 = lambda * self.power_m2 + delta * (p - self.power_mean_w);
        if let Some(fan) = fan_pct {
            self.fan_mean_pct = Some(match self.fan_mean_pct {
                Some(mean) => mean + (fan - mean) / (self.samples as f64).min(500.0),
                None => fan,
            });
        }
    }

    fn a(&self) -> f64 {
        self.theta[0]
    }

    pub fn tau_s(&self) -> f64 {
        let a = self.a();
        if a > 0.0 && a < 1.0 { -self.window_s / a.ln() } else { f64::INFINITY }
    }

    pub fn t_amb_c(&self) -> f64 {
        T_CENTRE + T_SCALE * self.theta[1] / (1.0 - self.a())
    }

    /// `R_ij` in °C/W.
    pub fn r_c_per_w(&self, j: usize) -> f64 {
        self.theta[2 + j] * T_SCALE / (P_SCALE * (1.0 - self.a()))
    }

    pub fn power_std_w(&self) -> f64 {
        if self.power_weight > 0.0 { (self.power_m2.max(0.0) / self.power_weight).sqrt() } else { 0.0 }
    }

    pub fn plausible(&self) -> bool {
        let a = self.a();
        if !(a > 0.0 && a < 1.0) {
            return false;
        }
        let tau = self.tau_s();
        let r = self.r_c_per_w(self.gpu);
        let t_amb = self.t_amb_c();
        (TAU_RANGE.0..=TAU_RANGE.1).contains(&tau)
            && (R_SELF_RANGE.0..=R_SELF_RANGE.1).contains(&r)
            && (T_AMB_RANGE.0..=T_AMB_RANGE.1).contains(&t_amb)
    }

    pub fn summary(&self, fit: &FitSettings) -> ModelSummary {
        let sample_part = if fit.min_samples == 0 { 1.0 } else { self.samples as f64 / fit.min_samples as f64 };
        let power_part = if fit.min_power_std_w <= 0.0 { 1.0 } else { self.power_std_w() / fit.min_power_std_w };
        let confidence = sample_part.min(power_part).clamp(0.0, 1.0);
        let residual_c = self.residual_ms.max(0.0).sqrt();
        let plausible = self.plausible();
        ModelSummary {
            r_self_c_per_w: self.r_c_per_w(self.gpu),
            r_c_per_w: (0..self.n_gpus).map(|j| self.r_c_per_w(j)).collect(),
            tau_s: self.tau_s(),
            t_amb_c: self.t_amb_c(),
            samples: self.samples,
            power_std_w: self.power_std_w(),
            residual_c,
            fan_mean_pct: self.fan_mean_pct,
            confidence,
            plausible,
            confident: plausible && confidence >= 1.0 && residual_c <= fit.max_residual_c,
        }
    }

    /// Predicted steady-state temperature with this GPU at `power_w` and the others at
    /// `others` (indexed by GPU; this GPU's own entry is ignored). Negative cross
    /// coupling is treated as none, which can only raise the prediction.
    pub fn steady_state_c(&self, power_w: f64, others: &[f64]) -> f64 {
        self.t_amb_c() + self.r_c_per_w(self.gpu) * power_w + self.cross_c(others)
    }

    fn cross_c(&self, others: &[f64]) -> f64 {
        (0..self.n_gpus).filter(|&j| j != self.gpu).map(|j| self.r_c_per_w(j).max(0.0) * others[j]).sum()
    }

    /// The highest power whose predicted steady state is at or below `target_c`.
    pub fn safe_power_w(&self, target_c: f64, others: &[f64]) -> f64 {
        (target_c - self.t_amb_c() - self.cross_c(others)) / self.r_c_per_w(self.gpu)
    }

    /// Predicted temperature after `t_s` seconds from `now_c` at constant powers.
    pub fn predict_c(&self, now_c: f64, power_w: f64, others: &[f64], t_s: f64) -> f64 {
        self.predict_scaled_c(now_c, power_w, others, t_s, 1.0)
    }

    /// As [`predict_c`](Self::predict_c), with the heating (R terms) scaled by `gain`.
    pub fn predict_scaled_c(&self, now_c: f64, power_w: f64, others: &[f64], t_s: f64, gain: f64) -> f64 {
        let steady = self.t_amb_c() + gain * (self.steady_state_c(power_w, others) - self.t_amb_c());
        steady + (now_c - steady) * (-t_s / self.tau_s()).exp()
    }

    /// The highest power whose predicted temperature stays at or below `target_c` over
    /// the next `horizon_s`, starting from `now_c`. The trajectory is monotone toward the
    /// steady state, so it binds at the horizon: `T(H) = S + (now − S)·e^(−H/τ) <= target`.
    /// Below target this allows a cool card more than its steady-state power (it has
    /// headroom to spend), and the allowance shrinks as it warms toward target; above
    /// target the card must be back at target by the horizon.
    pub fn planned_power_w(&self, target_c: f64, now_c: f64, others: &[f64], horizon_s: f64) -> f64 {
        self.planned_scaled_power_w(target_c, now_c, others, horizon_s, 1.0)
    }

    /// As [`planned_power_w`](Self::planned_power_w), with the heating scaled by `gain`.
    pub fn planned_scaled_power_w(&self, target_c: f64, now_c: f64, others: &[f64], horizon_s: f64, gain: f64) -> f64 {
        let decay = (-horizon_s / self.tau_s()).exp();
        let steady_allowed = (target_c - now_c * decay) / (1.0 - decay).max(1e-9);
        let rise_allowed = (steady_allowed - self.t_amb_c()) / gain;
        (rise_allowed - self.cross_c(others)) / self.r_c_per_w(self.gpu)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Groups 1 Hz readings into fit windows and feeds the model.
#[derive(Clone, Debug, Default)]
pub struct WindowAccumulator {
    elapsed_s: f64,
    power_sum: Vec<f64>,
    fan_sum: f64,
    fan_count: u32,
    last_temp: Option<f64>,
    /// Temperature at the end of the previous complete window.
    prev_end: Option<f64>,
    broken: bool,
}

/// A completed window ready to fit.
pub struct Window {
    pub t_prev: f64,
    pub t_next: f64,
    pub powers: Vec<f64>,
    pub fan_pct: Option<f64>,
}

impl WindowAccumulator {
    /// Add one sample; `None` when any GPU's reading failed (the window is discarded).
    pub fn push(&mut self, sample: Option<(f64, &[f64], Option<f64>)>, dt: f64, window_s: f64) -> Option<Window> {
        let Some((temp, powers, fan)) = sample else {
            self.broken = true;
            self.prev_end = None;
            return None;
        };
        if self.power_sum.len() != powers.len() {
            self.power_sum = vec![0.0; powers.len()];
        }
        for (sum, p) in self.power_sum.iter_mut().zip(powers) {
            *sum += p * dt;
        }
        if let Some(fan) = fan {
            self.fan_sum += fan;
            self.fan_count += 1;
        }
        self.elapsed_s += dt;
        self.last_temp = Some(temp);
        if self.elapsed_s + 1e-6 < window_s {
            return None;
        }
        let elapsed = self.elapsed_s;
        let window = (!self.broken).then(|| {
            self.prev_end.map(|t_prev| Window {
                t_prev,
                t_next: temp,
                powers: self.power_sum.iter().map(|s| s / elapsed).collect(),
                fan_pct: (self.fan_count > 0).then(|| self.fan_sum / f64::from(self.fan_count)),
            })
        });
        self.prev_end = Some(temp);
        self.broken = false;
        self.elapsed_s = 0.0;
        self.power_sum.iter_mut().for_each(|s| *s = 0.0);
        self.fan_sum = 0.0;
        self.fan_count = 0;
        window.flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fit() -> FitSettings {
        FitSettings { window_s: 5.0, forgetting: 0.999, min_samples: 100, min_power_std_w: 40.0, max_residual_c: 1.5 }
    }

    #[test]
    fn prior_maps_back_to_physical_values() {
        let model = ThermalModel::new(0, 2, 5.0);
        assert!((model.tau_s() - PRIOR_TAU_S).abs() < 1e-6);
        assert!((model.t_amb_c() - PRIOR_T_AMB).abs() < 1e-6);
        assert!((model.r_c_per_w(0) - PRIOR_R_SELF).abs() < 1e-9);
        assert!((model.r_c_per_w(1) - PRIOR_R_CROSS).abs() < 1e-9);
        assert!(!model.summary(&fit()).confident);
    }

    #[test]
    fn recovers_exact_discrete_plant() {
        // Noise-free plant in exactly the model's form.
        let (tau, t_amb, r_self, r_cross, dt): (f64, f64, f64, f64, f64) = (90.0, 28.0, 0.11, 0.025, 5.0);
        let a: f64 = (-dt / tau).exp();
        let mut model = ThermalModel::new(1, 2, dt);
        let mut t = 40.0;
        for k in 0..2000u32 {
            let p_self = [120.0, 450.0, 300.0, 600.0, 200.0][((k / 17) % 5) as usize];
            let p_other = [100.0, 500.0, 250.0][((k / 29) % 3) as usize];
            let next = a * t + (1.0 - a) * (t_amb + r_self * p_self + r_cross * p_other);
            model.update(t, next, &[p_other, p_self], Some(80.0), &fit());
            t = next;
        }
        let s = model.summary(&fit());
        assert!((s.tau_s - tau).abs() < 1.0, "{s:?}");
        assert!((s.r_self_c_per_w - r_self).abs() < 1e-3, "{s:?}");
        assert!((s.r_c_per_w[0] - r_cross).abs() < 1e-3, "{s:?}");
        assert!((s.t_amb_c - t_amb).abs() < 0.3, "{s:?}");
        assert!(s.confident, "{s:?}");
        let safe = model.safe_power_w(75.0, &[200.0, 0.0]);
        assert!((model.steady_state_c(safe, &[200.0, 0.0]) - 75.0).abs() < 1e-9);
    }

    #[test]
    fn windows_skip_failed_readings() {
        let mut acc = WindowAccumulator::default();
        let p = [100.0, 200.0];
        assert!(acc.push(Some((40.0, &p, None)), 1.0, 2.0).is_none());
        assert!(acc.push(Some((41.0, &p, None)), 1.0, 2.0).is_none(), "first window has no start");
        assert!(acc.push(Some((41.0, &p, None)), 1.0, 2.0).is_none());
        let w = acc.push(Some((42.0, &p, Some(50.0))), 1.0, 2.0).expect("second window");
        assert_eq!((w.t_prev, w.t_next, w.powers.clone(), w.fan_pct), (41.0, 42.0, vec![100.0, 200.0], Some(50.0)));
        assert!(acc.push(None, 1.0, 2.0).is_none());
        assert!(acc.push(Some((43.0, &p, None)), 1.0, 2.0).is_none());
        assert!(acc.push(Some((43.0, &p, None)), 1.0, 2.0).is_none(), "restart after a failure");
        assert!(acc.push(Some((43.0, &p, None)), 1.0, 2.0).is_none());
        assert!(acc.push(Some((44.0, &p, None)), 1.0, 2.0).is_some());
    }
}
