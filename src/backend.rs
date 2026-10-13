//! The device interface the controller drives: NVML on a real host, a simulated
//! plant in tests (`crate::sim`).
use serde::Serialize;
use std::fmt;

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

/// One sample of the fast, cheap sensors (about 0.1 ms each on an RTX PRO 6000).
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct Reading {
    pub temperature_c: f64,
    pub power_w: f64,
    pub utilization_pct: f64,
    /// The limit the board enforces right now.
    pub limit_w: f64,
    /// Average fan speed (percent of maximum); `None` on fanless boards or when unreadable.
    pub fan_pct: Option<f64>,
    /// The board reports a software power-cap clock event (it is power-limited);
    /// `None` when the driver does not report clock event reasons.
    pub power_capped: Option<bool>,
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
}
