//! The control law and learning against a simulated two-GPU thermal plant, through
//! the same `Daemon` code that drives NVML, with fault injection.
use gpu_powerd::config::{Config, Settings};
use gpu_powerd::control::Bound;
use gpu_powerd::daemon::Daemon;
use gpu_powerd::model::ThermalModel;
use gpu_powerd::sim::{Plant, SimBackend};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

/// Nothing crosses this in any scenario.
const HARD_C: f64 = 85.0;

struct Rig {
    plant: Rc<RefCell<Plant>>,
    daemon: Daemon<SimBackend>,
    /// Highest true (unquantised) temperature per GPU over every `run`.
    peak: [f64; 2],
}

fn settings(text: &str, plant: &Plant) -> Settings {
    let mut config = Config::parse(text).unwrap();
    config.state_file = PathBuf::new();
    config.model_file = PathBuf::new();
    config.stream = false;
    config.resolve(&plant.info).unwrap()
}

impl Rig {
    fn new(plant: Plant, config: &str) -> Self {
        let settings = settings(config, &plant);
        let plant = Rc::new(RefCell::new(plant));
        let daemon = Daemon::new(SimBackend::new(plant.clone()), settings);
        Self { plant, daemon, peak: [f64::MIN; 2] }
    }

    /// Run `seconds` one-second periods: control, then the plant advances under the new limits.
    fn run(&mut self, seconds: u32) {
        for _ in 0..seconds {
            self.daemon.tick(1.0).expect("tick");
            self.plant.borrow_mut().advance(1.0);
            let p = self.plant.borrow();
            for (i, g) in p.gpus.iter().enumerate() {
                self.peak[i] = self.peak[i].max(g.temp_c);
            }
        }
    }

    fn limit(&self, gpu: usize) -> f64 {
        self.plant.borrow().gpus[gpu].limit_w
    }

    fn temp(&self, gpu: usize) -> f64 {
        self.plant.borrow().gpus[gpu].temp_c
    }

    fn demand(&self, gpu: usize, watts: f64) {
        self.plant.borrow_mut().gpus[gpu].demand_w = watts;
    }

    fn bound(&self, gpu: usize) -> Bound {
        self.daemon.last.as_ref().unwrap().gpus[gpu].plan.bound
    }

    fn reload(&mut self, text: &str) {
        let plant = self.plant.borrow().clone();
        self.daemon.reload(settings(text, &plant));
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            for (i, peak) in self.peak.iter().enumerate() {
                assert!(*peak < HARD_C, "GPU {i} crossed {HARD_C} C: {peak:.2}");
            }
        }
    }
}

/// A well-cooled pair: 600 W settles at about 60 C, so thermal limits stay out of the way.
fn cool_plant() -> Plant {
    let mut plant = Plant::two_gpus();
    plant.r = vec![vec![0.05, 0.005], vec![0.005, 0.05]];
    plant
}

/// GPU 1 has poor airflow: it would settle near 100 C at 600 W.
fn hot_plant() -> Plant {
    let mut plant = Plant::two_gpus();
    plant.r = vec![vec![0.05, 0.012], vec![0.012, 0.12]];
    plant
}

fn confident(mut model: ThermalModel) -> ThermalModel {
    model.samples = 100_000;
    model.power_weight = 1.0;
    model.power_m2 = 100.0 * 100.0;
    model.residual_ms = 0.0;
    model
}

/// Confident models of `hot_plant` with GPU 1's self-heating scaled by `r_scale`.
fn hot_models(r_scale: f64) -> Vec<ThermalModel> {
    vec![
        confident(ThermalModel::with_params(0, 5.0, 110.0, 28.0, &[0.05, 0.012])),
        confident(ThermalModel::with_params(1, 5.0, 140.0, 28.0, &[0.012, 0.12 * r_scale])),
    ]
}

/// Full load on both cards for `seconds`: GPU 1's peak and its mean |T − 80| and mean
/// draw over the last half.
fn sustained(rig: &mut Rig, seconds: u32) -> (f64, f64, f64) {
    rig.demand(0, 700.0);
    rig.demand(1, 700.0);
    let (mut error, mut power, mut count) = (0.0, 0.0, 0.0);
    let mut peak = f64::MIN;
    for t in 0..seconds {
        rig.run(1);
        peak = peak.max(rig.temp(1));
        if t >= seconds / 2 {
            error += (rig.temp(1) - 80.0).abs();
            power += rig.plant.borrow().draw_w(1);
            count += 1.0;
        }
    }
    (peak, error / count, power / count)
}

#[test]
fn one_busy_card_gets_its_max_within_the_budget() {
    let mut rig = Rig::new(cool_plant(), "system_budget_w = 900\nlearn = false");
    rig.demand(0, 700.0);
    rig.run(20);
    assert_eq!(rig.limit(0), 600.0, "busy card at its own max");
    assert_eq!(rig.limit(1), 300.0, "idle card: base plus the spare budget");
    assert_eq!(rig.bound(0), Bound::Max);
    let gpu1 = &rig.daemon.last.as_ref().unwrap().gpus[1].plan;
    assert!(gpu1.idle && gpu1.demand_w == 150.0, "idle demand is base: {gpu1:?}");

    // Without idle_takes_spare the idle card stays at base.
    rig.reload("system_budget_w = 900\nlearn = false\nidle_takes_spare = false");
    rig.run(5);
    assert_eq!((rig.limit(0), rig.limit(1)), (600.0, 150.0));

    // Both idle: an even split of the budget.
    rig.reload("system_budget_w = 900\nlearn = false");
    rig.demand(0, 0.0);
    rig.run(20);
    assert_eq!((rig.limit(0), rig.limit(1)), (450.0, 450.0));
}

#[test]
fn two_busy_cards_split_the_budget() {
    let mut rig = Rig::new(cool_plant(), "system_budget_w = 900\nlearn = false");
    rig.demand(0, 700.0);
    rig.demand(1, 700.0);
    rig.run(30);
    assert_eq!((rig.limit(0), rig.limit(1)), (450.0, 450.0), "equal demand: equal split");
    assert_eq!(rig.bound(0), Bound::Budget);

    // Uneven: GPU 1 only wants 250 W, so GPU 0 gets the rest up to its max.
    rig.demand(1, 250.0);
    rig.run(30);
    // GPU 1's demand settles just above 300 W, so GPU 0's 599.x W quantises to 595.
    assert!(rig.limit(0) >= 595.0, "GPU 0 near its max: {}", rig.limit(0));
    assert!((295.0..=305.0).contains(&rig.limit(1)), "GPU 1 keeps its draw plus headroom: {}", rig.limit(1));

    // GPU 1 picks up again: the split returns to even, and the budget always holds.
    rig.demand(1, 700.0);
    for _ in 0..60 {
        rig.run(1);
        assert!(rig.limit(0) + rig.limit(1) <= 900.0 + 1e-9);
    }
    assert_eq!((rig.limit(0), rig.limit(1)), (450.0, 450.0));
}

#[test]
fn a_hot_card_is_held_at_target_and_its_budget_moves_to_the_other() {
    let mut rig = Rig::new(hot_plant(), "system_budget_w = 900\nlearn = false");
    rig.demand(0, 700.0);
    rig.demand(1, 700.0);
    rig.run(30);
    assert_eq!((rig.limit(0), rig.limit(1)), (450.0, 450.0), "cool start: even split");
    rig.run(1800);
    let (l0, l1) = (rig.limit(0), rig.limit(1));
    assert!(l1 < 420.0, "hot GPU 1 held by the PI controller: {l1}");
    assert!(l0 > 480.0, "GPU 0 took the freed budget: {l0}");
    assert!(l0 + l1 <= 900.0);
    assert!((rig.temp(1) - 80.0).abs() < 1.0, "GPU 1 held at target: {}", rig.temp(1));
    assert!(rig.temp(0) < 75.0, "GPU 0 stays cool: {}", rig.temp(0));
    assert_eq!(rig.bound(1), Bound::Pi);
}

#[test]
fn pi_alone_settles_at_target_without_crossing_hard() {
    let mut rig = Rig::new(hot_plant(), "system_budget_w = 1200\nlearn = false");
    let (peak, error, _) = sustained(&mut rig, 3600);
    assert!(peak < HARD_C, "peak {peak}");
    assert!(error < 0.5, "settles at about 80 C: mean |T-80| {error:.2}");
    assert_eq!(rig.bound(1), Bound::Pi);
}

#[test]
fn a_confident_model_tracks_target_tighter_than_pi() {
    let mut pi = Rig::new(hot_plant(), "system_budget_w = 1200\nlearn = false");
    let (pi_peak, pi_error, pi_power) = sustained(&mut pi, 3600);
    let mut model = Rig::new(hot_plant(), "system_budget_w = 1200\nlearn = false");
    model.daemon.set_models(hot_models(1.0));
    let (m_peak, m_error, m_power) = sustained(&mut model, 3600);
    eprintln!(
        "GPU 1 sustained full load: PI peak {pi_peak:.2} C, |T-80| {pi_error:.2}, {pi_power:.0} W; \
         model peak {m_peak:.2} C, |T-80| {m_error:.2}, {m_power:.0} W"
    );
    assert!(pi_peak > 82.0, "PI alone overshoots target: {pi_peak}");
    assert!(m_peak <= 80.6, "the model does not overshoot: {m_peak}");
    assert!(m_error < 1.0, "the model settles at target: {m_error}");
    // Holding 0.5 C under target (5 W quantisation, rounded down) costs under 3% power.
    assert!(m_power > 0.97 * pi_power, "little wasted headroom: {m_power} vs {pi_power}");
    assert!(matches!(model.bound(1), Bound::Model | Bound::Correction));
}

#[test]
fn the_model_reduces_overshoot_on_bursts() {
    // Bursts of full load: GPU 1's overshoot above target and its degree-seconds above it.
    fn bursts(rig: &mut Rig) -> (f64, f64) {
        let (mut peak, mut area) = (0.0f64, 0.0);
        for _ in 0..3 {
            rig.demand(0, 700.0);
            rig.demand(1, 700.0);
            for _ in 0..900 {
                rig.run(1);
                peak = peak.max(rig.temp(1) - 80.0);
                area += (rig.temp(1) - 80.0).max(0.0);
            }
            rig.demand(0, 0.0);
            rig.demand(1, 0.0);
            rig.run(900);
        }
        (peak, area)
    }
    let mut pi = Rig::new(hot_plant(), "system_budget_w = 1200\nlearn = false");
    let (pi_peak, pi_area) = bursts(&mut pi);
    let mut model = Rig::new(hot_plant(), "system_budget_w = 1200\nlearn = false");
    model.daemon.set_models(hot_models(1.0));
    let (m_peak, m_area) = bursts(&mut model);
    eprintln!("GPU 1 above 80 C: PI peak {pi_peak:.2} C, {pi_area:.0} C*s; model peak {m_peak:.2} C, {m_area:.0} C*s");
    assert!(pi_peak > 2.0);
    assert!(m_peak < 1.0);
    assert!(m_area < 0.1 * pi_area);
}

#[test]
fn a_cool_card_runs_at_its_full_share_with_no_early_cap() {
    // The cool card's model is confident and correct: it never caps a card that cannot
    // reach target, at any point.
    let mut rig = Rig::new(cool_plant(), "system_budget_w = 1200\nlearn = false");
    let cool = |gpu| confident(ThermalModel::with_params(gpu, 5.0, 110.0, 28.0, &[0.05, 0.005]));
    rig.daemon.set_models(vec![cool(0), cool(1)]);
    rig.demand(0, 700.0);
    rig.demand(1, 700.0);
    for _ in 0..1800 {
        rig.run(1);
        assert_eq!((rig.limit(0), rig.limit(1)), (600.0, 600.0));
    }
    // And on the hot plant, the model lets GPU 1 run at 600 W while it has headroom.
    let mut rig = Rig::new(hot_plant(), "system_budget_w = 1200\nlearn = false");
    rig.daemon.set_models(hot_models(1.0));
    rig.demand(1, 700.0);
    rig.run(30);
    assert_eq!(rig.limit(1), 600.0, "a cold card is not capped early: {}", rig.limit(1));
}

#[test]
fn a_wrong_model_is_corrected_before_hard() {
    for scale in [0.5, 2.0] {
        let mut rig = Rig::new(hot_plant(), "system_budget_w = 1200\nlearn = false");
        rig.daemon.set_models(hot_models(scale));
        let (peak, error, _) = sustained(&mut rig, 3600);
        let gain = rig.daemon.last.as_ref().unwrap().gpus[1].plan.model_gain;
        eprintln!("R x{scale}: peak {peak:.2} C, |T-80| {error:.2}, learned gain {gain:.2}");
        assert!(peak < 82.0, "R x{scale}: corrected well before hard_c: {peak}");
        assert!(error < 1.0, "R x{scale}: settles at target: {error}");
        assert!((gain * scale - 1.0).abs() < 0.25, "R x{scale}: the gain undoes the error: {gain}");
    }
}

#[test]
fn model_may_raise_false_keeps_the_model_at_or_below_pi() {
    let mut rig = Rig::new(hot_plant(), "system_budget_w = 1200\nlearn = false\nmodel_may_raise = false");
    rig.daemon.set_models(hot_models(1.0));
    rig.demand(1, 700.0);
    for _ in 0..1800 {
        rig.run(1);
        let plan = &rig.daemon.last.as_ref().unwrap().gpus[1].plan;
        assert!(plan.thermal_cap_w <= plan.pi_w + 1e-6, "{plan:?}");
    }
}

#[test]
fn hard_c_cuts_to_base_at_once_and_raises_slowly() {
    let mut rig = Rig::new(cool_plant(), "learn = false");
    rig.demand(0, 700.0);
    rig.demand(1, 700.0);
    rig.run(10);
    assert_eq!(rig.limit(0), 600.0);
    rig.plant.borrow_mut().gpus[0].fake_temp_c = Some(85.0);
    rig.run(1);
    assert_eq!(rig.limit(0), 150.0, "hard_c: straight to base in one period");
    assert_eq!(rig.bound(0), Bound::Hard);
    assert_eq!(rig.limit(1), 600.0, "the other card is unaffected");
    rig.plant.borrow_mut().gpus[0].fake_temp_c = None;
    let mut previous = rig.limit(0);
    for _ in 0..30 {
        rig.run(1);
        let now = rig.limit(0);
        assert!(now - previous <= 10.0 + 1e-9, "raises slowly: {previous} -> {now}");
        previous = now;
    }
    assert!(previous < 600.0, "still recovering after 30 s: {previous}");
    rig.run(60);
    assert_eq!(rig.limit(0), 600.0, "recovered");
}

#[test]
fn a_sensor_failure_puts_that_card_at_sensor_fail_w() {
    let mut rig = Rig::new(cool_plant(), "system_budget_w = 900\nlearn = false");
    rig.demand(0, 700.0);
    rig.demand(1, 700.0);
    rig.run(20);
    assert_eq!((rig.limit(0), rig.limit(1)), (450.0, 450.0));

    rig.plant.borrow_mut().gpus[1].fail_read = Some(15); // NVML_ERROR_GPU_IS_LOST
    rig.run(1);
    assert_eq!(rig.limit(1), 150.0, "failed reading: base (the default sensor_fail_w) at once");
    assert_eq!(rig.bound(1), Bound::Sensor);
    rig.run(10);
    assert_eq!(rig.limit(0), 600.0, "the healthy card may use the freed budget");

    // Implausible readings count as failures too.
    rig.plant.borrow_mut().gpus[1].fail_read = None;
    rig.plant.borrow_mut().gpus[1].fake_temp_c = Some(255.0);
    rig.run(1);
    assert_eq!(rig.limit(1), 150.0);
    assert_eq!(rig.bound(1), Bound::Sensor);

    // Readings return: the card recovers, slowly, back to an even split.
    rig.plant.borrow_mut().gpus[1].fake_temp_c = None;
    rig.run(1);
    assert!(rig.limit(1) <= 160.0, "no jump on recovery: {}", rig.limit(1));
    rig.run(60);
    assert_eq!((rig.limit(0), rig.limit(1)), (450.0, 450.0));

    // sensor_fail_w is configurable per card.
    rig.reload("system_budget_w = 900\nlearn = false\n[gpu.1]\nsensor_fail_w = 200");
    rig.plant.borrow_mut().gpus[1].fail_read = Some(15);
    rig.run(1);
    assert_eq!(rig.limit(1), 200.0);
}

#[test]
fn a_config_reload_applies_the_new_values() {
    let mut rig = Rig::new(cool_plant(), "system_budget_w = 900\nlearn = false");
    rig.demand(0, 700.0);
    rig.demand(1, 700.0);
    rig.run(20);
    assert_eq!((rig.limit(0), rig.limit(1)), (450.0, 450.0));

    // Lower GPU 0's max: it is cut at once; GPU 1 grows into the freed budget.
    rig.reload("system_budget_w = 900\nlearn = false\n[gpu.0]\nmax_w = 300");
    rig.run(1);
    assert_eq!(rig.limit(0), 300.0);
    rig.run(10);
    assert_eq!(rig.limit(1), 600.0);

    // A larger budget and a lower target: the cool plant now runs hot enough to be held.
    rig.reload("system_budget_w = 1200\nlearn = false\nthrottle_start_c = 45\ntarget_c = 50\nhard_c = 70");
    rig.run(1800);
    assert!((rig.temp(0) - 50.0).abs() < 1.0, "held at the new target: {}", rig.temp(0));
    assert!(rig.limit(0) > 300.0, "the old max_w no longer applies: {}", rig.limit(0));
}

#[test]
fn every_key_round_trips_through_toml_and_reload() {
    // Every key, with a value other than its default.
    let text = r#"
        system_budget_w = 1000
        base_w = 160
        max_w = 590
        fallback_w = 300
        sensor_fail_w = 170
        throttle_start_c = 70
        target_c = 78
        hard_c = 84
        pi_kp_w_per_c = 20
        pi_ki_w_per_c_s = 0.5
        correction_kp_w_per_c = 30
        correction_rate_per_s = 0.1
        use_model = false
        model_may_raise = false
        period_s = 0.5
        fan_read_s = 20
        tlimit_read_s = 0
        process_read_s = 30
        step_w = 10
        reassert_s = 60
        smoothing_s = 2
        raise_w_per_s = 5
        budget_raise_w_per_s = 50
        pinned_fraction = 0.9
        idle_util_pct = 10
        demand_headroom_w = 40
        idle_takes_spare = false
        learn = false
        model_horizon_s = 90
        model_window_s = 10
        model_forgetting = 0.995
        model_min_samples = 100
        model_min_power_std_w = 30
        model_max_residual_c = 2
        model_save_s = 600
        state_file = "/tmp/gpu-powerd-test/state.json"
        model_file = "/tmp/gpu-powerd-test/model.json"
        stream = false
        stream_socket = "/tmp/gpu-powerd-test/stream.sock"
        websocket = false
        websocket_bind = "127.0.0.1"
        websocket_port = 9999
        [gpu."E1:00.0"]
        base_w = 170
        max_w = 450
        fallback_w = 320
        sensor_fail_w = 180
        throttle_start_c = 72
        target_c = 77
        hard_c = 82
        pi_kp_w_per_c = 15
        pi_ki_w_per_c_s = 0.7
        correction_kp_w_per_c = 35
        correction_rate_per_s = 0.2
        use_model = true
        model_may_raise = true
    "#;
    let config = Config::parse(text).unwrap();
    // TOML out and back in is the same config.
    let again = Config::parse(&toml::to_string(&config).unwrap()).unwrap();
    assert_eq!(config, again);
    // No key was ignored: every one differs from its default after resolving.
    let plant = cool_plant();
    let resolved = config.resolve(&plant.info).unwrap();
    let defaults = Config::default().resolve(&plant.info).unwrap();
    let card = |s: &Settings, i: usize| serde_json::to_value(&s.gpus[i]).unwrap();
    // GPU 0 takes every top-level card key; GPU 1 overrides every one of them again.
    let (g0, g1, d) = (card(&resolved, 0), card(&resolved, 1), card(&defaults, 0));
    for (key, value) in g0.as_object().unwrap() {
        assert_ne!(value, &d[key], "GPU 0: {key} kept its default");
        assert_ne!(value, &g1[key], "GPU 1: [gpu] override of {key} not applied");
    }
    let top = |s: &Settings| {
        let mut v = serde_json::to_value(&s.config).unwrap();
        let o = v.as_object_mut().unwrap();
        o.remove("gpu");
        for k in serde_json::to_value(&s.gpus[0]).unwrap().as_object().unwrap().keys() {
            o.remove(k);
        }
        v
    };
    let (a, b) = (top(&resolved), top(&defaults));
    for (key, value) in a.as_object().unwrap() {
        assert_ne!(value, &b[key], "{key} kept its default");
    }
    // Through a reload (the SIGHUP path): the daemon runs on the new values.
    let mut rig = Rig::new(cool_plant(), "learn = false");
    rig.run(3);
    let mut reloaded = config.clone();
    reloaded.state_file = PathBuf::new();
    reloaded.model_file = PathBuf::new();
    let settings = reloaded.resolve(&rig.plant.borrow().info).unwrap();
    rig.daemon.reload(settings.clone());
    assert_eq!(rig.daemon.settings, settings);
    rig.run(5);
    let state = rig.daemon.last.as_ref().unwrap();
    assert_eq!(state.gpus[1].max_w, 450.0);
    assert_eq!(state.gpus[1].target_c, 77.0);
    assert_eq!(state.gpus[0].target_c, 78.0);
    assert_eq!(state.budget_w, 1000.0);
    assert!(rig.plant.borrow().writes.iter().rev().take(4).all(|(_, w)| w % 10 == 0), "step_w = 10 applies");
}

#[test]
fn an_invalid_reload_keeps_the_running_config() {
    // The daemon only ever receives validated settings: Config::parse/resolve reject a
    // bad file, and the run loop then keeps the running config (main.rs, Event::Reload).
    let plant = cool_plant();
    for bad in [
        "target_c = 90",
        "throttle_start_c = 81",
        "fallback_w = 100",
        "fallback_w = 700",
        "base_w = 400\nfallback_w = 300",
        "sensor_fail_w = 50",
    ] {
        assert!(Config::parse(bad).and_then(|c| c.resolve(&plant.info)).is_err(), "{bad:?}");
    }
}

/// A deterministic workload: each GPU picks a demand (idle or 200..700 W) every 20-120 s.
fn varied_workload(rig: &mut Rig, seconds: u32, seed: u64) {
    let mut state = seed;
    let mut next = |bound: u64| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) % bound
    };
    let mut change = [0u32; 2];
    for t in 0..seconds {
        for (gpu, at) in change.iter_mut().enumerate() {
            if t >= *at {
                let demand = if next(4) == 0 { 0.0 } else { 200.0 + next(500) as f64 };
                rig.demand(gpu, demand);
                *at = t + 20 + next(100) as u32;
            }
        }
        rig.run(1);
    }
}

#[test]
fn the_model_converges_in_simulation() {
    // The model learns while the PI controller holds GPU 1 (which spends time at target,
    // so its power varies with the controller as well as the workload).
    let mut rig = Rig::new(hot_plant(), "use_model = false");
    varied_workload(&mut rig, 4 * 3600, 7);
    let fit = gpu_powerd::daemon::fit_settings(&rig.daemon.settings.config);
    let plant = rig.plant.borrow().clone();
    for gpu in 0..2 {
        let s = rig.daemon.models()[gpu].summary(&fit);
        let tau = plant.gpus[gpu].tau_s;
        let r_self = plant.r[gpu][gpu];
        let r_cross = plant.r[gpu][1 - gpu];
        eprintln!("GPU {gpu}: fitted {s:?}; true tau {tau}, R self {r_self}, R cross {r_cross}");
        assert!(s.confident, "GPU {gpu} confident: {s:?}");
        assert!((s.r_self_c_per_w - r_self).abs() / r_self < 0.1, "R self {} vs {r_self}", s.r_self_c_per_w);
        assert!((s.r_c_per_w[1 - gpu] - r_cross).abs() < 0.006, "R cross {} vs {r_cross}", s.r_c_per_w[1 - gpu]);
        assert!((s.tau_s - tau).abs() / tau < 0.2, "tau {} vs {tau}", s.tau_s);
        assert!((s.t_amb_c - plant.t_amb_c).abs() < 2.5, "T_amb {} vs {}", s.t_amb_c, plant.t_amb_c);
    }
}

#[test]
fn learn_then_lead_end_to_end() {
    // Learn on a varied workload under PI, then let the learned model lead: sustained
    // full load stays at target with no overshoot.
    let mut rig = Rig::new(hot_plant(), "system_budget_w = 1200\nuse_model = false");
    varied_workload(&mut rig, 4 * 3600, 11);
    let models = rig.daemon.models().to_vec();
    let start = rig.plant.borrow().clone();
    let mut led = Rig::new(start, "system_budget_w = 1200\nlearn = false");
    led.daemon.set_models(models);
    led.demand(0, 0.0);
    led.demand(1, 0.0);
    led.run(1800);
    let (peak, error, _) = sustained(&mut led, 3600);
    eprintln!("learned model leading: peak {peak:.2} C, |T-80| {error:.2}");
    assert!(peak < 81.0, "{peak}");
    assert!(error < 1.0, "{error}");
}

#[test]
fn a_failed_write_leads_to_fallback() {
    let mut rig = Rig::new(cool_plant(), "fallback_w = 325\nlearn = false");
    rig.demand(0, 700.0);
    rig.run(5);
    assert_eq!(rig.limit(0), 600.0);
    // Force a change (hard temperature) and fail the write.
    rig.plant.borrow_mut().gpus[0].fake_temp_c = Some(90.0);
    rig.plant.borrow_mut().fail_writes = 1;
    let error = rig.daemon.tick(1.0).expect_err("the write fails");
    assert!(error.to_string().contains("simulated write failure"), "{error:#}");
    // The run loop returns the error, which drops the daemon: every GPU goes to fallback.
    let plant = rig.plant.clone();
    drop(rig);
    let p = plant.borrow();
    assert_eq!((p.gpus[0].limit_w, p.gpus[1].limit_w), (325.0, 325.0));
    assert_eq!(&p.writes[p.writes.len() - 2..], &[(0, 325), (1, 325)]);
}

#[test]
fn a_panic_still_restores_the_fallback() {
    let plant = Rc::new(RefCell::new(cool_plant()));
    let config = settings("fallback_w = 200\nlearn = false", &plant.borrow());
    let shared = plant.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let mut daemon = Daemon::new(SimBackend::new(shared), config);
        daemon.tick(1.0).unwrap();
        panic!("simulated crash");
    }));
    assert!(result.is_err());
    let p = plant.borrow();
    assert_eq!((p.gpus[0].limit_w, p.gpus[1].limit_w), (200.0, 200.0));
}

#[test]
fn limits_are_written_on_change_and_reasserted_every_30_s() {
    let mut rig = Rig::new(cool_plant(), "learn = false");
    rig.demand(0, 700.0);
    rig.run(5);
    let writes = rig.plant.borrow().writes.len();
    rig.run(25);
    assert_eq!(rig.plant.borrow().writes.len(), writes, "no rewrites while unchanged");
    rig.run(1);
    assert_eq!(rig.plant.borrow().writes.len(), writes + 2, "both GPUs reasserted at 30 s");
    assert!(rig.plant.borrow().writes.iter().all(|(_, w)| w % 5 == 0));
}

#[test]
fn a_fast_period_is_supported() {
    // period_s = 0.1: ten samples a second; rates are per second, so the outcome matches.
    let mut rig = Rig::new(hot_plant(), "system_budget_w = 1200\nlearn = false\nperiod_s = 0.1");
    rig.demand(1, 700.0);
    for _ in 0..36000 {
        rig.daemon.tick(0.1).unwrap();
        rig.plant.borrow_mut().advance(0.1);
        rig.peak[1] = rig.peak[1].max(rig.temp(1));
    }
    assert!((rig.temp(1) - 80.0).abs() < 1.0, "{}", rig.temp(1));
}

#[test]
fn state_and_model_files_round_trip() {
    let dir = std::env::temp_dir().join(format!("gpu-powerd-state-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let plant = hot_plant();
    let text = format!(
        "state_file = {:?}\nmodel_file = {:?}\nstream = false\n",
        dir.join("state.json").display().to_string(),
        dir.join("model.json").display().to_string()
    );
    let settings = Config::parse(&text).unwrap().resolve(&plant.info).unwrap();
    let plant = Rc::new(RefCell::new(plant));
    let mut rig = Rig {
        plant: plant.clone(),
        daemon: Daemon::new(SimBackend::new(plant.clone()), settings.clone()),
        peak: [f64::MIN; 2],
    };
    varied_workload(&mut rig, 600, 3);
    let models = rig.daemon.models().to_vec();
    let state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("state.json")).unwrap()).unwrap();
    assert_eq!(state["running"], true);
    assert_eq!(state["gpus"].as_array().unwrap().len(), 2);
    for key in ["limit_w", "plan", "reading", "model", "throttle_reasons", "target_c"] {
        assert!(!state["gpus"][0][key].is_null(), "state has {key}");
    }
    for key in [
        "sm_clock_mhz",
        "mem_clock_mhz",
        "mem_utilization_pct",
        "fan_pct",
        "tlimit_c",
        "mem_total_mib",
        "mem_used_mib",
        "mem_free_mib",
        "mem_reserved_mib",
    ] {
        assert!(!state["gpus"][1]["reading"][key].is_null(), "reading has {key}");
    }
    assert!(state["gpus"][1]["model"]["r_self_c_per_w"].is_number());
    drop(rig);
    let state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("state.json")).unwrap()).unwrap();
    assert_eq!(state["running"], false, "exit marks the state stopped");
    let daemon = Daemon::new(SimBackend::new(plant.clone()), settings);
    assert_eq!(daemon.models(), &models[..]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_sample_is_published_to_the_stream() {
    let plant = cool_plant();
    let mut rig = Rig::new(plant, "learn = false");
    let lines = Rc::new(RefCell::new(Vec::<String>::new()));
    let sink = lines.clone();
    rig.daemon.publish =
        Some(Box::new(move |line: &[u8]| sink.borrow_mut().push(String::from_utf8_lossy(line).into())));
    rig.demand(0, 700.0);
    for _ in 0..20 {
        rig.daemon.tick(0.1).unwrap();
        rig.plant.borrow_mut().advance(0.1);
    }
    let lines = lines.borrow();
    assert_eq!(lines.len(), 20, "one message per sample");
    let message: serde_json::Value = serde_json::from_str(lines.last().unwrap().trim_end()).unwrap();
    let gpu = &message["gpus"][0];
    for key in [
        "temperature_c",
        "power_w",
        "limit_w",
        "sm_clock_mhz",
        "mem_clock_mhz",
        "utilization_pct",
        "mem_utilization_pct",
        "fan_pct",
        "tlimit_c",
        "mem_total_mib",
        "mem_used_mib",
        "mem_free_mib",
        "mem_reserved_mib",
    ] {
        assert!(gpu["reading"][key].is_number(), "stream reading has {key}");
    }
    assert!(gpu["processes"].is_array(), "stream carries the process list");
    assert_eq!(gpu["processes"][0]["pid"], 4242);
    assert!(gpu["throttle_reasons"].is_array());
    assert!(gpu["plan"]["bound"].is_string() && gpu["plan"]["reason"].is_string());
    assert!(gpu["model"]["confident"].is_boolean());
}
