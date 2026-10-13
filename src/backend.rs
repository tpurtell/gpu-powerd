//! The device interface the controller drives: NVML on a real host, a simulated
//! plant in tests (`crate::sim`).
use serde::Serialize;
use std::fmt;
use std::time::Duration;

/// Static facts about one GPU, read once at startup.
#[derive(Clone, Debug, Serialize)]
pub struct DeviceInfo {
    /// NVML index.
    pub index: u32,
    pub uuid: String,
    pub name: String,
    /// PCI bus id, e.g. `00000000:E1:00.0`.
    pub pci: String,
    /// The board's power-limit range (`nvmlDeviceGetPowerManagementLimitConstraints`).
    pub board_min_w: f64,
    pub board_max_w: f64,
    /// The board's default power limit, when the driver reports it.
    pub default_w: Option<f64>,
}

/// One sample. The core fields are cheap reads (microseconds); `fan_pct` and
/// `tlimit_c` come from firmware RPCs and are refreshed at their own slow cadence.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct Reading {
    pub temperature_c: f64,
    pub power_w: f64,
    pub utilization_pct: f64,
    /// Memory controller utilization (percent).
    pub mem_utilization_pct: f64,
    /// The limit the board enforces right now.
    pub limit_w: f64,
    pub sm_clock_mhz: Option<f64>,
    pub mem_clock_mhz: Option<f64>,
    /// NVML clock event (throttle) reason bits; see [`throttle_reasons`].
    pub throttle_mask: Option<u64>,
    /// Average fan speed (percent of maximum); `None` on fanless boards or when unreadable.
    pub fan_pct: Option<f64>,
    /// T.Limit margin: degrees below the GPU's maximum operating temperature.
    pub tlimit_c: Option<f64>,
    /// The board reports a software power-cap clock event (it is power-limited).
    pub power_capped: Option<bool>,
    /// VRAM (MiB) from `nvmlDeviceGetMemoryInfo[_v2]`; `None` when unreadable.
    pub mem_total_mib: Option<f64>,
    pub mem_used_mib: Option<f64>,
    pub mem_free_mib: Option<f64>,
    /// Memory the driver reserves (only the `_v2` call reports it).
    pub mem_reserved_mib: Option<f64>,
}

/// One compute process on a GPU (`nvmlDeviceGetComputeRunningProcesses_v3`).
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ProcessUse {
    pub pid: u32,
    /// `None` where NVML reports N/A (WSL, some containers).
    pub used_mib: Option<f64>,
}

impl Reading {
    /// A reading with only the fields the controller needs (tests and simulations).
    pub fn basic(temperature_c: f64, power_w: f64, utilization_pct: f64, limit_w: f64) -> Self {
        Self {
            temperature_c,
            power_w,
            utilization_pct,
            mem_utilization_pct: 0.0,
            limit_w,
            sm_clock_mhz: None,
            mem_clock_mhz: None,
            throttle_mask: None,
            fan_pct: None,
            tlimit_c: None,
            power_capped: None,
            mem_total_mib: None,
            mem_used_mib: None,
            mem_free_mib: None,
            mem_reserved_mib: None,
        }
    }
}

/// NVML `nvmlClocksEventReason*` bits and their names.
pub const THROTTLE_REASONS: [(u64, &str); 9] = [
    (0x1, "gpu_idle"),
    (0x2, "applications_clocks_setting"),
    (0x4, "sw_power_cap"),
    (0x8, "hw_slowdown"),
    (0x10, "sync_boost"),
    (0x20, "sw_thermal_slowdown"),
    (0x40, "hw_thermal_slowdown"),
    (0x80, "hw_power_brake_slowdown"),
    (0x100, "display_clock_setting"),
];

/// The names of the active reasons in an NVML clock event mask.
pub fn throttle_reasons(mask: u64) -> Vec<&'static str> {
    THROTTLE_REASONS.iter().filter(|(bit, _)| mask & bit != 0).map(|(_, name)| *name).collect()
}

/// A failed device call.
#[derive(Clone, Debug, PartialEq)]
pub struct BackendError {
    pub call: &'static str,
    /// NVML return code (0 for failures outside NVML).
    pub code: i32,
    pub message: String,
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.call, self.message)
    }
}

impl std::error::Error for BackendError {}

/// What the controller needs from the GPUs.
pub trait Backend {
    fn devices(&self) -> &[DeviceInfo];
    fn read(&mut self, gpu: usize) -> Result<Reading, BackendError>;
    fn set_limit_w(&mut self, gpu: usize, watts: u32) -> Result<(), BackendError>;
    /// How often to refresh the slow (firmware RPC) readings: fan speed, and the
    /// T.Limit margin (`None`: never).
    fn set_slow_cadence(&mut self, _fan: Duration, _tlimit: Option<Duration>, _process: Option<Duration>) {}
    /// The compute processes on a GPU, cached and refreshed at the process cadence;
    /// `None` when the list is off or unreadable.
    fn processes(&mut self, _gpu: usize) -> Option<Vec<ProcessUse>> {
        None
    }
}
