//! A simulated multi-GPU thermal plant behind the [`Backend`] interface, with fault
//! injection, for tests and offline experiments. Not used by the daemon itself.
//!
//! Plant: `τ_i·dT_i/dt = T_amb + Σ_j R_ij·P_j − T_i`. Each GPU's workload asks for
//! `demand_w` (0 = idle); it draws `min(demand, limit)`, or `idle_w` when idle.
//! Readings are quantised like NVML's: whole degrees, milliwatts, whole percent.
use crate::backend::{Backend, BackendError, DeviceInfo, ProcessUse, Reading};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct SimGpu {
    pub tau_s: f64,
    pub temp_c: f64,
    pub limit_w: f64,
    pub demand_w: f64,
    pub idle_w: f64,
    /// Fault injection: reads fail with this NVML code.
    pub fail_read: Option<i32>,
    /// Fault injection: reads report this temperature instead.
    pub fake_temp_c: Option<f64>,
    /// The draw over the last step (NVML's power reading is a one-second average).
    pub last_draw_w: f64,
}

#[derive(Clone, Debug)]
pub struct Plant {
    pub t_amb_c: f64,
    /// `r[i][j]`: °C at GPU i per watt at GPU j.
    pub r: Vec<Vec<f64>>,
    pub gpus: Vec<SimGpu>,
    pub info: Vec<DeviceInfo>,
    /// Fault injection: the next this many limit writes fail.
    pub fail_writes: usize,
    /// Every successful write, as (gpu, watts).
    pub writes: Vec<(usize, u32)>,
    pub time_s: f64,
}

impl Plant {
    /// Two RTX PRO 6000-like boards (150..600 W) in one case. GPU 1 runs hotter
    /// (worse airflow) and each heats the other a little.
    pub fn two_gpus() -> Self {
        let info = (0..2)
            .map(|i| DeviceInfo {
                index: i,
                uuid: format!("GPU-5133000{i}-0000-4000-8000-000000000000"),
                name: "Simulated RTX PRO 6000".into(),
                pci: format!("00000000:{}:00.0", if i == 0 { "11" } else { "E1" }),
                board_min_w: 150.0,
                board_max_w: 600.0,
                default_w: Some(600.0),
            })
            .collect();
        let gpu = |tau_s| SimGpu {
            tau_s,
            temp_c: 32.0,
            limit_w: 600.0,
            demand_w: 0.0,
            idle_w: 30.0,
            fail_read: None,
            fake_temp_c: None,
            last_draw_w: 30.0,
        };
        Self {
            t_amb_c: 28.0,
            r: vec![vec![0.085, 0.012], vec![0.012, 0.100]],
            gpus: vec![gpu(110.0), gpu(140.0)],
            info,
            fail_writes: 0,
            writes: Vec::new(),
            time_s: 0.0,
        }
    }

    pub fn draw_w(&self, i: usize) -> f64 {
        let g = &self.gpus[i];
        if g.demand_w <= 0.0 { g.idle_w } else { g.demand_w.min(g.limit_w).max(g.idle_w) }
    }

    pub fn steady_state_c(&self, i: usize, powers: &[f64]) -> f64 {
        self.t_amb_c + self.r[i].iter().zip(powers).map(|(r, p)| r * p).sum::<f64>()
    }

    /// Advance the plant by `dt` (exact for powers held over the step).
    pub fn advance(&mut self, dt: f64) {
        let powers: Vec<f64> = (0..self.gpus.len()).map(|i| self.draw_w(i)).collect();
        for i in 0..self.gpus.len() {
            let target = self.steady_state_c(i, &powers);
            let g = &mut self.gpus[i];
            g.last_draw_w = powers[i];
            g.temp_c = target + (g.temp_c - target) * (-dt / g.tau_s).exp();
        }
        self.time_s += dt;
    }

    fn reading(&self, i: usize) -> Reading {
        let g = &self.gpus[i];
        let draw = g.last_draw_w;
        let temp = g.fake_temp_c.unwrap_or(g.temp_c.round());
        let busy = g.demand_w > 0.0;
        let capped = busy && draw >= g.limit_w - 0.5;
        Reading {
            temperature_c: temp,
            power_w: (draw * 1000.0).round() / 1000.0,
            utilization_pct: if busy { 100.0 } else { 0.0 },
            mem_utilization_pct: if busy { 60.0 } else { 0.0 },
            limit_w: g.limit_w,
            sm_clock_mhz: Some(if busy { 2400.0 } else { 210.0 }),
            mem_clock_mhz: Some(if busy { 14000.0 } else { 405.0 }),
            throttle_mask: Some(if capped {
                0x4
            } else if busy {
                0
            } else {
                0x1
            }),
            fan_pct: Some((30.0 + (g.temp_c - 35.0).max(0.0) * 2.5).min(100.0).round()),
            tlimit_c: Some(90.0 - g.temp_c.round()),
            power_capped: Some(capped),
            mem_total_mib: Some(97887.0),
            mem_used_mib: Some(if busy { 24000.0 } else { 600.0 }),
            mem_free_mib: Some(if busy { 97887.0 - 24000.0 - 500.0 } else { 97887.0 - 600.0 - 500.0 }),
            mem_reserved_mib: Some(500.0),
        }
    }
}

/// A [`Backend`] over a shared [`Plant`] (the test keeps a handle to drive it).
pub struct SimBackend {
    pub plant: Rc<RefCell<Plant>>,
    info: Vec<DeviceInfo>,
}

impl SimBackend {
    pub fn new(plant: Rc<RefCell<Plant>>) -> Self {
        let info = plant.borrow().info.clone();
        Self { plant, info }
    }
}

impl Backend for SimBackend {
    fn devices(&self) -> &[DeviceInfo] {
        &self.info
    }

    fn read(&mut self, gpu: usize) -> Result<Reading, BackendError> {
        let plant = self.plant.borrow();
        if let Some(code) = plant.gpus[gpu].fail_read {
            return Err(BackendError {
                call: "nvmlDeviceGetTemperature",
                code,
                message: format!("simulated failure ({code})"),
            });
        }
        Ok(plant.reading(gpu))
    }

    fn processes(&mut self, gpu: usize) -> Option<Vec<ProcessUse>> {
        let busy = self.plant.borrow().gpus[gpu].demand_w > 0.0;
        Some(if busy { vec![ProcessUse { pid: 4242, used_mib: Some(23900.0) }] } else { vec![] })
    }

    fn set_limit_w(&mut self, gpu: usize, watts: u32) -> Result<(), BackendError> {
        let mut plant = self.plant.borrow_mut();
        if plant.fail_writes > 0 {
            plant.fail_writes -= 1;
            return Err(BackendError {
                call: "nvmlDeviceSetPowerManagementLimit",
                code: 999,
                message: "simulated write failure".into(),
            });
        }
        let info = &plant.info[gpu];
        let w = f64::from(watts);
        if w < info.board_min_w || w > info.board_max_w {
            return Err(BackendError {
                call: "nvmlDeviceSetPowerManagementLimit",
                code: 2,
                message: "Invalid Argument".into(),
            });
        }
        plant.gpus[gpu].limit_w = w;
        plant.writes.push((gpu, watts));
        Ok(())
    }
}
