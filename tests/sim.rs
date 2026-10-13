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

struct Rig {
    plant: Rc<RefCell<Plant>>,
    daemon: Daemon<SimBackend>,
}

fn settings(text: &str, plant: &Plant) -> Settings {
    let mut config = Config::parse(text).unwrap();
    config.state_file = PathBuf::new();
    config.model_file = PathBuf::new();
    config.resolve(&plant.info).unwrap()
}

impl Rig {
    fn new(plant: Plant, config: &str) -> Self {
        let settings = settings(config, &plant);
        let plant = Rc::new(RefCell::new(plant));
        let daemon = Daemon::new(SimBackend::new(plant.clone()), settings);
        Self { plant, daemon }
    }

    /// Run `seconds` one-second periods: control, then the plant advances under the new limits.
    fn run(&mut self, seconds: u32) {
        for _ in 0..seconds {
            self.daemon.tick(1.0).expect("tick");
            self.plant.borrow_mut().advance(1.0);
        }
    }

    /// Like `run`, but returns the highest true (unquantised) temperature per GPU.
    fn run_peak(&mut self, seconds: u32) -> [f64; 2] {
        let mut peak = [f64::MIN; 2];
        for _ in 0..seconds {
            self.daemon.tick(1.0).expect("tick");
            self.plant.borrow_mut().advance(1.0);
            let p = self.plant.borrow();
            for (i, g) in p.gpus.iter().enumerate() {
                peak[i] = peak[i].max(g.temp_c);
            }
        }
        peak
    }

    fn limit(&self, gpu: usize) -> f64 {
        self.plant.borrow().gpus[gpu].limit_w
    }

    fn demand(&self, gpu: usize, watts: f64) {
        self.plant.borrow_mut().gpus[gpu].demand_w = watts;
    }

    fn bound(&self, gpu: usize) -> Bound {
        self.daemon.last.as_ref().unwrap().gpus[gpu].plan.bound
    }
}

/// A well-cooled pair: 600 W settles at about 60 C, so thermal limits stay out of the way.
fn cool_plant() -> Plant {
    let mut plant = Plant::two_gpus();
    plant.r = vec![vec![0.05, 0.005], vec![0.005, 0.05]];
    plant
}

/// GPU 1 has poor airflow (like PCI E1 on raptor): above 75 C well before 600 W.
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

    // With a tighter budget the idle card has exactly base.
    let plant = rig.plant.borrow().clone();
    rig.daemon.reload(settings("system_budget_w = 750\nlearn = false", &plant));
    rig.run(5);
    assert_eq!((rig.limit(0), rig.limit(1)), (600.0, 150.0));
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
    assert!(rig.limit(0) + rig.limit(1) <= 900.0);

    // GPU 1 picks up again: the split returns to even, and the budget always holds.
    rig.demand(1, 700.0);
    for _ in 0..60 {
        rig.run(1);
        assert!(rig.limit(0) + rig.limit(1) <= 900.0 + 1e-9);
    }
    assert_eq!((rig.limit(0), rig.limit(1)), (450.0, 450.0));
}

#[test]
fn unlimited_budget_lets_both_cards_reach_max() {
    let mut rig = Rig::new(cool_plant(), "system_budget_w = 1200\nlearn = false");
    rig.demand(0, 700.0);
    rig.demand(1, 700.0);
    rig.run(20);
    assert_eq!((rig.limit(0), rig.limit(1)), (600.0, 600.0));
}

#[test]
fn a_hot_card_is_capped_and_its_budget_moves_to_the_other() {
    let mut rig = Rig::new(hot_plant(), "system_budget_w = 900\nlearn = false");
    rig.demand(0, 700.0);
    rig.demand(1, 700.0);
    rig.run(30);
    assert_eq!((rig.limit(0), rig.limit(1)), (450.0, 450.0), "cool start: even split");
    rig.run(1800);
    let (l0, l1) = (rig.limit(0), rig.limit(1));
    let p = rig.plant.borrow();
    assert!(l1 < 420.0, "hot GPU 1 capped by the ramp: {l1}");
    assert!(l0 > 480.0, "GPU 0 took the freed budget: {l0}");
    assert!(l0 + l1 <= 900.0);
    assert!(p.gpus[1].temp_c > 75.0 && p.gpus[1].temp_c < 85.0, "GPU 1 held on the ramp: {}", p.gpus[1].temp_c);
    assert!(p.gpus[0].temp_c < 75.0, "GPU 0 stays below its own thermal limit: {}", p.gpus[0].temp_c);
    drop(p);
    assert_eq!(rig.bound(1), Bound::Ramp);
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
    // Smoothed temperature decays first (still >= hard for a moment), then the cap rises at 10 W/s.
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
fn a_sensor_failure_puts_that_card_at_base() {
    let mut rig = Rig::new(cool_plant(), "system_budget_w = 900\nlearn = false");
    rig.demand(0, 700.0);
    rig.demand(1, 700.0);
    rig.run(20);
    assert_eq!((rig.limit(0), rig.limit(1)), (450.0, 450.0));

    rig.plant.borrow_mut().gpus[1].fail_read = Some(15); // NVML_ERROR_GPU_IS_LOST
    rig.run(1);
    assert_eq!(rig.limit(1), 150.0, "failed reading: base at once");
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
}

#[test]
fn a_config_reload_applies_the_new_values() {
    let mut rig = Rig::new(cool_plant(), "system_budget_w = 900\nlearn = false");
    rig.demand(0, 700.0);
    rig.demand(1, 700.0);
    rig.run(20);
    assert_eq!((rig.limit(0), rig.limit(1)), (450.0, 450.0));

    // Lower GPU 0's max: it is cut at once; GPU 1 grows into the freed budget.
    let plant = rig.plant.borrow().clone();
    rig.daemon.reload(settings("system_budget_w = 900\nlearn = false\n[gpu.0]\nmax_w = 300", &plant));
    rig.run(1);
    assert_eq!(rig.limit(0), 300.0);
    rig.run(10);
    assert_eq!(rig.limit(1), 600.0);

    // A larger budget and a lower throttle start.
    rig.daemon.reload(settings("system_budget_w = 1200\nlearn = false\nthrottle_start_c = 40\nhard_c = 90", &plant));
    rig.run(600);
    let p = rig.plant.borrow();
    let ramp = |t: f64| 600.0 - 450.0 * (t - 40.0) / 50.0;
    assert!(p.gpus[0].temp_c > 40.0);
    assert!(rig.limit(0) <= ramp(p.gpus[0].temp_c.round() - 1.0) + 5.0, "the new ramp governs: {}", rig.limit(0));
    assert!(rig.limit(0) > 300.0, "the old max_w no longer applies: {}", rig.limit(0));
}

#[test]
fn the_model_only_lowers_limits() {
    let mut rig = Rig::new(cool_plant(), "learn = false");
    rig.demand(0, 700.0);
    rig.demand(1, 700.0);
    rig.run(10);
    assert_eq!((rig.limit(0), rig.limit(1)), (600.0, 600.0));

    // A model that predicts a very cool card (it would allow far above max) changes nothing.
    let cool = |gpu| confident(ThermalModel::with_params(gpu, 5.0, 100.0, 20.0, &[0.001, 0.001]));
    rig.daemon.set_models(vec![cool(0), cool(1)]);
    rig.run(30);
    assert_eq!((rig.limit(0), rig.limit(1)), (600.0, 600.0), "never above max_w");

    // ...nor does it lift a ramp cut: with GPU 0 reading 80 C the ramp (375 W) still holds.
    rig.plant.borrow_mut().gpus[0].fake_temp_c = Some(80.0);
    rig.run(30);
    assert!(rig.limit(0) <= 375.0, "the ramp still governs: {}", rig.limit(0));
    assert_eq!(rig.bound(0), Bound::Ramp);
    rig.plant.borrow_mut().gpus[0].fake_temp_c = None;
    rig.run(120);

    // A model that predicts a hot card lowers the limit before it heats up.
    let hot = confident(ThermalModel::with_params(1, 5.0, 100.0, 28.0, &[0.012, 0.12]));
    rig.daemon.set_models(vec![cool(0), hot]);
    rig.run(1);
    let expected = (74.0 - 28.0 - 0.012 * 600.0) / 0.12;
    assert!((rig.limit(1) - expected).abs() <= 6.0, "model limit {} vs {expected}", rig.limit(1));
    assert_eq!(rig.bound(1), Bound::Model);
    assert!(rig.plant.borrow().gpus[1].temp_c < 75.0, "acted before the card heated");
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
        // Predicted steady state at full load within a degree or two of the plant's.
        let predicted = rig.daemon.models()[gpu].steady_state_c(450.0, &[450.0, 450.0]);
        let truth = plant.steady_state_c(gpu, &[450.0, 450.0]);
        assert!((predicted - truth).abs() < 2.0, "steady state {predicted} vs {truth}");
    }
}

/// Bursts of full load on both cards: returns GPU 1's peak overshoot above 75 C and
/// its degree-seconds above 75 C.
fn bursts(rig: &mut Rig) -> (f64, f64) {
    let (mut peak, mut area) = (0.0f64, 0.0);
    for _ in 0..3 {
        rig.demand(0, 700.0);
        rig.demand(1, 700.0);
        for _ in 0..900 {
            let p = rig.run_peak(1);
            peak = peak.max(p[1] - 75.0);
            area += (rig.plant.borrow().gpus[1].temp_c - 75.0).max(0.0);
        }
        rig.demand(0, 0.0);
        rig.demand(1, 0.0);
        rig.run(900);
    }
    (peak, area)
}

#[test]
fn the_learned_model_overshoots_less_than_the_ramp_alone() {
    // Learn on a varied workload (the model only watches), then replay the same bursts
    // with the ramp alone and with the model acting.
    let mut rig = Rig::new(hot_plant(), "system_budget_w = 1200\nuse_model = false");
    varied_workload(&mut rig, 4 * 3600, 11);
    let models = rig.daemon.models().to_vec();
    let fit = gpu_powerd::daemon::fit_settings(&rig.daemon.settings.config);
    assert!(models.iter().all(|m| m.summary(&fit).confident));

    let start = rig.plant.borrow().clone();
    let mut ramp_only = Rig::new(start.clone(), "system_budget_w = 1200\nuse_model = false\nlearn = false");
    let (ramp_peak, ramp_area) = bursts(&mut ramp_only);
    let mut with_model = Rig::new(start, "system_budget_w = 1200\nlearn = false");
    with_model.daemon.set_models(models);
    let (model_peak, model_area) = bursts(&mut with_model);
    eprintln!(
        "GPU 1 above 75 C: ramp only peak {ramp_peak:.2} C, {ramp_area:.0} C*s; model peak {model_peak:.2} C, {model_area:.0} C*s"
    );
    assert!(ramp_peak > 3.0, "the ramp alone settles well above throttle_start: {ramp_peak}");
    assert!(model_peak < 1.0, "the model holds the card at throttle_start: {model_peak}");
    assert!(model_area < 0.1 * ramp_area);
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
    let Rig { plant, daemon } = rig;
    drop(daemon);
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
    // Every write is a multiple of 5 W.
    assert!(rig.plant.borrow().writes.iter().all(|(_, w)| w % 5 == 0));
}

#[test]
fn state_and_model_files_round_trip() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("state_and_model_files_round_trip");
    let _ = std::fs::remove_dir_all(&dir);
    let plant = hot_plant();
    let text = format!(
        "state_file = {:?}\nmodel_file = {:?}\n",
        dir.join("state.json").display().to_string(),
        dir.join("model.json").display().to_string()
    );
    let settings = Config::parse(&text).unwrap().resolve(&plant.info).unwrap();
    let plant = Rc::new(RefCell::new(plant));
    let mut rig = Rig { plant: plant.clone(), daemon: Daemon::new(SimBackend::new(plant.clone()), settings.clone()) };
    varied_workload(&mut rig, 600, 3);
    let models = rig.daemon.models().to_vec();
    let state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("state.json")).unwrap()).unwrap();
    assert_eq!(state["running"], true);
    assert_eq!(state["gpus"].as_array().unwrap().len(), 2);
    for key in ["limit_w", "plan", "reading", "model"] {
        assert!(!state["gpus"][0][key].is_null(), "state has {key}");
    }
    assert!(state["gpus"][1]["model"]["r_self_c_per_w"].is_number());
    drop(rig);
    let state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("state.json")).unwrap()).unwrap();
    assert_eq!(state["running"], false, "exit marks the state stopped");
    // A restart picks the saved model up.
    let daemon = Daemon::new(SimBackend::new(plant.clone()), settings);
    assert_eq!(daemon.models(), &models[..]);
}
