//! Time each NVML call the control loop makes (read-only): `cargo run --example nvml_timing`.
use gpu_powerd::backend::Backend;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut nvml = gpu_powerd::nvml::Nvml::open()?;
    let n = nvml.devices().len();
    for round in 0..5 {
        if round > 0 {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        let started = Instant::now();
        for i in 0..n {
            nvml.read(i).map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        println!(
            "control-period read {round} (fans, process list cached after the first; includes the VRAM read): {:.3} ms per GPU",
            started.elapsed().as_secs_f64() * 1000.0 / n as f64
        );
    }
    println!("back to back (20 rounds):");
    for (name, ms) in nvml.time_calls(0, 20) {
        println!("  {name}: {ms:.3} ms");
    }
    for gpu in 0..n {
        std::thread::sleep(std::time::Duration::from_secs(1));
        println!("GPU {gpu}, once after a 1 s pause (the control cadence):");
        for (name, ms) in nvml.time_calls(gpu, 1) {
            println!("  {name}: {ms:.3} ms");
        }
    }
    Ok(())
}
