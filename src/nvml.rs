//! NVML in-process, through `dlopen("libnvidia-ml.so.1")`: no link-time dependency
//! on the driver, initialised once, never spawning `nvidia-smi`.
//!
//! Per-period reads are the cheap ones (temperature, power, utilization, enforced
//! limit, SM and memory clocks, clock event reasons; microseconds each on an RTX PRO
//! 6000 with driver 595). Fan speed is a GSP RPC (about 1-7 ms per fan, measured with
//! `--example nvml_timing`), and so is the T.Limit margin: both are read at their own
//! slow cadence (`fan_read_s`, `tlimit_read_s`) and cached between. The cumulative
//! throttle counters are never read.
use crate::backend::{Backend, BackendError, DeviceInfo, Reading};
use anyhow::{Context, Result, bail};
use libloading::os::unix::{Library, Symbol};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uint, c_ulonglong, c_void};

type Device = *mut c_void;
type Ret = c_int;
const SUCCESS: Ret = 0;
/// Default cadence for the firmware-RPC readings.
pub const FAN_PERIOD: std::time::Duration = std::time::Duration::from_secs(10);
const CLOCK_SM: c_uint = 1;
const CLOCK_MEM: c_uint = 2;

/// `nvmlMarginTemperature_v1_t`.
#[repr(C)]
struct Margin {
    version: c_uint,
    margin: c_int,
}
const MARGIN_V1: c_uint = std::mem::size_of::<Margin>() as c_uint | (1 << 24);
const TEMPERATURE_GPU: c_uint = 0;
/// `nvmlClocksEventReasonSwPowerCap`.
const REASON_SW_POWER_CAP: c_ulonglong = 0x4;

/// `nvmlPciInfo_t` (`nvmlDeviceGetPciInfo_v3`).
#[repr(C)]
struct PciInfo {
    bus_id_legacy: [c_char; 16],
    domain: c_uint,
    bus: c_uint,
    device: c_uint,
    pci_device_id: c_uint,
    pci_sub_system_id: c_uint,
    bus_id: [c_char; 32],
}

/// `nvmlUtilization_t`.
#[repr(C)]
#[derive(Default)]
struct Utilization {
    gpu: c_uint,
    memory: c_uint,
}

type VoidFn = unsafe extern "C" fn() -> Ret;
type ErrorStringFn = unsafe extern "C" fn(Ret) -> *const c_char;
type CountFn = unsafe extern "C" fn(*mut c_uint) -> Ret;
type HandleFn = unsafe extern "C" fn(c_uint, *mut Device) -> Ret;
type StringFn = unsafe extern "C" fn(Device, *mut c_char, c_uint) -> Ret;
type PciFn = unsafe extern "C" fn(Device, *mut PciInfo) -> Ret;
type UintFn = unsafe extern "C" fn(Device, *mut c_uint) -> Ret;
type TemperatureFn = unsafe extern "C" fn(Device, c_uint, *mut c_uint) -> Ret;
type ConstraintsFn = unsafe extern "C" fn(Device, *mut c_uint, *mut c_uint) -> Ret;
type UtilizationFn = unsafe extern "C" fn(Device, *mut Utilization) -> Ret;
type ReasonsFn = unsafe extern "C" fn(Device, *mut c_ulonglong) -> Ret;
type FanFn = unsafe extern "C" fn(Device, c_uint, *mut c_uint) -> Ret;
type SetLimitFn = unsafe extern "C" fn(Device, c_uint) -> Ret;
type ClockFn = unsafe extern "C" fn(Device, c_uint, *mut c_uint) -> Ret;
type MarginFn = unsafe extern "C" fn(Device, *mut Margin) -> Ret;

struct Gpu {
    handle: Device,
    fans: u32,
    /// The last fan reading and when it was taken.
    fan: Option<(std::time::Instant, Option<f64>)>,
    /// The last T.Limit margin and when it was taken.
    tlimit: Option<(std::time::Instant, Option<f64>)>,
}

/// An initialised NVML session over every device the driver reports.
/// Dropping it calls `nvmlShutdown`.
pub struct Nvml {
    error_string: ErrorStringFn,
    shutdown: VoidFn,
    temperature: TemperatureFn,
    power: UintFn,
    limit: UintFn,
    utilization: UtilizationFn,
    set_limit: SetLimitFn,
    reasons: Option<ReasonsFn>,
    fan_speed: Option<FanFn>,
    clock: Option<ClockFn>,
    margin: Option<MarginFn>,
    fan_period: std::time::Duration,
    tlimit_period: Option<std::time::Duration>,
    gpus: Vec<Gpu>,
    info: Vec<DeviceInfo>,
    // Keeps every function pointer above valid; dropped last.
    _library: Library,
}

fn symbol<T: Copy>(library: &Library, name: &str) -> Option<T> {
    // SAFETY: each call site names the symbol with its exact nvml.h signature as `T`.
    unsafe { library.get::<T>(name.as_bytes()).ok().map(|s: Symbol<T>| *s) }
}

fn required<T: Copy>(library: &Library, name: &str) -> Result<T> {
    symbol(library, name).with_context(|| format!("libnvidia-ml.so.1 has no {name}"))
}

fn c_string(buffer: &[c_char]) -> String {
    // SAFETY: the buffer is zero-initialised and NVML writes a NUL-terminated string
    // within its length, so it is terminated even after a failed call.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_string_lossy().into_owned()
}

impl Nvml {
    /// Load and initialise NVML and enumerate its devices.
    pub fn open() -> Result<Self> {
        // SAFETY: loading the driver's NVML library runs only its own initialisers.
        let library = unsafe { Library::new("libnvidia-ml.so.1") }
            .context("dlopen libnvidia-ml.so.1 (is the NVIDIA driver installed?)")?;
        let init: VoidFn = required(&library, "nvmlInit_v2")?;
        let count: CountFn = required(&library, "nvmlDeviceGetCount_v2")?;
        let by_index: HandleFn = required(&library, "nvmlDeviceGetHandleByIndex_v2")?;
        let uuid: StringFn = required(&library, "nvmlDeviceGetUUID")?;
        let name: StringFn = required(&library, "nvmlDeviceGetName")?;
        let pci: Option<PciFn> = symbol(&library, "nvmlDeviceGetPciInfo_v3");
        let constraints: ConstraintsFn = required(&library, "nvmlDeviceGetPowerManagementLimitConstraints")?;
        let default_limit: Option<UintFn> = symbol(&library, "nvmlDeviceGetPowerManagementDefaultLimit");
        let num_fans: Option<UintFn> = symbol(&library, "nvmlDeviceGetNumFans");
        let mut nvml = Self {
            error_string: required(&library, "nvmlErrorString")?,
            shutdown: required(&library, "nvmlShutdown")?,
            temperature: required(&library, "nvmlDeviceGetTemperature")?,
            power: required(&library, "nvmlDeviceGetPowerUsage")?,
            limit: required(&library, "nvmlDeviceGetEnforcedPowerLimit")?,
            utilization: required(&library, "nvmlDeviceGetUtilizationRates")?,
            set_limit: required(&library, "nvmlDeviceSetPowerManagementLimit")?,
            reasons: symbol(&library, "nvmlDeviceGetCurrentClocksEventReasons")
                .or_else(|| symbol(&library, "nvmlDeviceGetCurrentClocksThrottleReasons")),
            fan_speed: symbol(&library, "nvmlDeviceGetFanSpeed_v2"),
            clock: symbol(&library, "nvmlDeviceGetClockInfo"),
            margin: symbol(&library, "nvmlDeviceGetMarginTemperature"),
            fan_period: FAN_PERIOD,
            tlimit_period: Some(FAN_PERIOD),
            gpus: Vec::new(),
            info: Vec::new(),
            _library: library,
        };
        // SAFETY: nvmlInit_v2 takes no arguments; the matching nvmlShutdown runs on drop.
        let status = unsafe { init() };
        if status != SUCCESS {
            let message = nvml.message(status);
            // Without a successful init there is nothing to shut down.
            std::mem::forget(nvml);
            bail!("nvmlInit_v2: {message}");
        }
        let mut devices: c_uint = 0;
        // SAFETY: a valid out-pointer to a c_uint.
        nvml.check("nvmlDeviceGetCount_v2", unsafe { count(&mut devices) })?;
        for index in 0..devices {
            let mut handle: Device = std::ptr::null_mut();
            // SAFETY: index < count; a valid out-pointer for the handle.
            nvml.check("nvmlDeviceGetHandleByIndex_v2", unsafe { by_index(index, &mut handle) })?;
            let mut uuid_buf = [0 as c_char; 96];
            // SAFETY: the buffer length is passed alongside it.
            nvml.check("nvmlDeviceGetUUID", unsafe { uuid(handle, uuid_buf.as_mut_ptr(), uuid_buf.len() as c_uint) })?;
            let mut name_buf = [0 as c_char; 96];
            // SAFETY: as above.
            nvml.check("nvmlDeviceGetName", unsafe { name(handle, name_buf.as_mut_ptr(), name_buf.len() as c_uint) })?;
            let pci_id = pci
                .and_then(|pci| {
                    // SAFETY: PciInfo is plain data; zeroed is a valid initial value.
                    let mut info: PciInfo = unsafe { std::mem::zeroed() };
                    // SAFETY: a valid out-pointer to nvmlPciInfo_t (v3 layout).
                    (unsafe { pci(handle, &mut info) } == SUCCESS).then(|| c_string(&info.bus_id))
                })
                .unwrap_or_default();
            let (mut low, mut high): (c_uint, c_uint) = (0, 0);
            // SAFETY: valid out-pointers.
            nvml.check("nvmlDeviceGetPowerManagementLimitConstraints", unsafe {
                constraints(handle, &mut low, &mut high)
            })?;
            let default_w = default_limit.and_then(|f| {
                let mut value: c_uint = 0;
                // SAFETY: a valid out-pointer.
                (unsafe { f(handle, &mut value) } == SUCCESS).then_some(f64::from(value) / 1000.0)
            });
            let fans = num_fans
                .and_then(|f| {
                    let mut value: c_uint = 0;
                    // SAFETY: a valid out-pointer.
                    (unsafe { f(handle, &mut value) } == SUCCESS).then_some(value)
                })
                .unwrap_or(0);
            nvml.gpus.push(Gpu { handle, fans, fan: None, tlimit: None });
            nvml.info.push(DeviceInfo {
                index,
                uuid: c_string(&uuid_buf),
                name: c_string(&name_buf),
                pci: pci_id,
                board_min_w: f64::from(low) / 1000.0,
                board_max_w: f64::from(high) / 1000.0,
                default_w,
            });
        }
        Ok(nvml)
    }

    fn message(&self, status: Ret) -> String {
        // SAFETY: nvmlErrorString returns a static string for any code (or null).
        let raw = unsafe { (self.error_string)(status) };
        if raw.is_null() {
            return format!("NVML error {status}");
        }
        // SAFETY: a non-null result is a static NUL-terminated string.
        format!("{} ({status})", unsafe { CStr::from_ptr(raw) }.to_string_lossy())
    }

    fn check(&self, call: &'static str, status: Ret) -> Result<(), BackendError> {
        if status == SUCCESS { Ok(()) } else { Err(BackendError { call, code: status, message: self.message(status) }) }
    }

    fn uint(&self, call: &'static str, f: UintFn, handle: Device) -> Result<c_uint, BackendError> {
        let mut value: c_uint = 0;
        // SAFETY: `f` has the (device, unsigned int*) signature; a valid out-pointer.
        self.check(call, unsafe { f(handle, &mut value) })?;
        Ok(value)
    }
}

impl Nvml {
    /// The T.Limit margin (°C below the maximum operating temperature).
    fn read_margin(&self, handle: Device) -> Option<f64> {
        self.margin.and_then(|f| {
            let mut margin = Margin { version: MARGIN_V1, margin: 0 };
            // SAFETY: a valid out-pointer to a versioned nvmlMarginTemperature_t.
            (unsafe { f(handle, &mut margin) } == SUCCESS).then_some(f64::from(margin.margin))
        })
    }

    /// Mean fan speed over every fan, or `None` when any read fails.
    fn read_fans(&self, handle: Device, fans: u32) -> Option<f64> {
        self.fan_speed.filter(|_| fans > 0).and_then(|f| {
            let mut total = 0.0;
            for fan in 0..fans {
                let mut speed: c_uint = 0;
                // SAFETY: fan < nvmlDeviceGetNumFans; a valid out-pointer.
                if unsafe { f(handle, fan, &mut speed) } != SUCCESS {
                    return None;
                }
                total += f64::from(speed);
            }
            Some(total / f64::from(fans))
        })
    }

    /// Mean time of each per-period call on one GPU over `rounds` (diagnostics; read-only).
    pub fn time_calls(&self, gpu: usize, rounds: u32) -> Vec<(&'static str, f64)> {
        let Gpu { handle, fans, .. } = self.gpus[gpu];
        let time = |f: &dyn Fn()| {
            let started = std::time::Instant::now();
            for _ in 0..rounds {
                f();
            }
            started.elapsed().as_secs_f64() * 1000.0 / f64::from(rounds)
        };
        let mut out = vec![];
        let mut value: c_uint = 0;
        let v = &mut value as *mut c_uint;
        // SAFETY (all below): valid handles and out-pointers that outlive each call.
        out.push((
            "temperature",
            time(&|| unsafe {
                (self.temperature)(handle, TEMPERATURE_GPU, v);
            }),
        ));
        out.push((
            "power usage",
            time(&|| unsafe {
                (self.power)(handle, v);
            }),
        ));
        out.push((
            "enforced limit",
            time(&|| unsafe {
                (self.limit)(handle, v);
            }),
        ));
        let mut u = Utilization::default();
        let up = &mut u as *mut Utilization;
        out.push((
            "utilization",
            time(&|| unsafe {
                (self.utilization)(handle, up);
            }),
        ));
        if let Some(f) = self.reasons {
            let mut r: c_ulonglong = 0;
            let rp = &mut r as *mut c_ulonglong;
            out.push((
                "clock event reasons",
                time(&|| unsafe {
                    f(handle, rp);
                }),
            ));
        }
        if let Some(f) = self.fan_speed.filter(|_| fans > 0) {
            out.push((
                "fan speed (per fan)",
                time(&|| unsafe {
                    f(handle, 0, v);
                }),
            ));
        }
        out
    }
}

impl Backend for Nvml {
    fn devices(&self) -> &[DeviceInfo] {
        &self.info
    }

    fn read(&mut self, gpu: usize) -> Result<Reading, BackendError> {
        let Gpu { handle, fans, fan, tlimit } = self.gpus[gpu];
        let mut temperature: c_uint = 0;
        // SAFETY: a valid device handle and out-pointer.
        self.check("nvmlDeviceGetTemperature", unsafe {
            (self.temperature)(handle, TEMPERATURE_GPU, &mut temperature)
        })?;
        let power_mw = self.uint("nvmlDeviceGetPowerUsage", self.power, handle)?;
        let limit_mw = self.uint("nvmlDeviceGetEnforcedPowerLimit", self.limit, handle)?;
        let mut utilization = Utilization::default();
        // SAFETY: a valid out-pointer to nvmlUtilization_t.
        self.check("nvmlDeviceGetUtilizationRates", unsafe { (self.utilization)(handle, &mut utilization) })?;
        let throttle_mask = self.reasons.and_then(|f| {
            let mut reasons: c_ulonglong = 0;
            // SAFETY: a valid out-pointer to an unsigned long long.
            (unsafe { f(handle, &mut reasons) } == SUCCESS).then_some(reasons)
        });
        let clock = |which: c_uint| {
            self.clock.and_then(|f| {
                let mut mhz: c_uint = 0;
                // SAFETY: a valid out-pointer; `which` is an nvmlClockType_t.
                (unsafe { f(handle, which, &mut mhz) } == SUCCESS).then_some(f64::from(mhz))
            })
        };
        let (sm_clock_mhz, mem_clock_mhz) = (clock(CLOCK_SM), clock(CLOCK_MEM));
        let now = std::time::Instant::now();
        let fan_pct = match fan.filter(|(at, _)| at.elapsed() < self.fan_period) {
            Some((_, cached)) => cached,
            None => {
                let value = self.read_fans(handle, fans);
                self.gpus[gpu].fan = Some((now, value));
                value
            }
        };
        let tlimit_c = match self.tlimit_period {
            None => None,
            Some(period) => match tlimit.filter(|(at, _)| at.elapsed() < period) {
                Some((_, cached)) => cached,
                None => {
                    let value = self.read_margin(handle);
                    self.gpus[gpu].tlimit = Some((now, value));
                    value
                }
            },
        };
        Ok(Reading {
            temperature_c: f64::from(temperature),
            power_w: f64::from(power_mw) / 1000.0,
            utilization_pct: f64::from(utilization.gpu),
            mem_utilization_pct: f64::from(utilization.memory),
            limit_w: f64::from(limit_mw) / 1000.0,
            sm_clock_mhz,
            mem_clock_mhz,
            throttle_mask,
            fan_pct,
            tlimit_c,
            power_capped: throttle_mask.map(|m| m & REASON_SW_POWER_CAP != 0),
        })
    }

    fn set_slow_cadence(&mut self, fan: std::time::Duration, tlimit: Option<std::time::Duration>) {
        self.fan_period = fan;
        self.tlimit_period = tlimit;
    }

    fn set_limit_w(&mut self, gpu: usize, watts: u32) -> Result<(), BackendError> {
        let handle = self.gpus[gpu].handle;
        // SAFETY: a valid device handle; the limit is passed by value in milliwatts.
        self.check("nvmlDeviceSetPowerManagementLimit", unsafe { (self.set_limit)(handle, watts * 1000) })
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        // SAFETY: init succeeded (open() forgets the struct otherwise); no handle outlives self.
        unsafe { (self.shutdown)() };
    }
}
