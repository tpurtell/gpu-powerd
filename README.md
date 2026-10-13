# gpu-powerd

A thermal- and budget-aware power-limit daemon for NVIDIA GPUs. Once a second it
sets each GPU's power limit from three things:

- **A shared power budget**, split by load. An idle card stays near its base; a busy
  card gets up to its own max; two busy cards split by demand (900 W becomes 450/450,
  or 600/300 when one wants less).
- **A temperature ramp** for each card: full power below `throttle_start_c`, falling
  linearly to `base_w` at `hard_c`, and straight to `base_w` at or above `hard_c`.
- **A learned thermal model** for each card. It predicts the steady-state temperature
  at a given power and lowers the limit *before* the card heats up, so the firmware's
  late, aggressive thermal throttle never has to act.

It replaces a fixed boot-time limit (`nvidia-smi -pl` in a oneshot unit). It does not
control fans; it records fan speed as an input to the model.

## Why

Some boards have no firmware temperature target, only a power limit. Under sustained
load (long LLM prefills, for example), a card with poor airflow can heat until it
falls off the PCIe bus (Xid 79). A fixed low limit prevents that, but costs
performance all the time, including on the cool card and in short bursts. gpu-powerd
lets each card run at full power while it is cool and the budget allows, and holds a
hot card at the power that keeps it below `throttle_start_c`.

## Safety rules

- **The model can only lower a limit, never raise it.** Each GPU's limit is
  `min(budget share, ramp(T_smoothed), model_safe_power, max_w)`, floored at
  `base_w`. The model acts only after it is confident: enough samples, enough power
  variation, a small fit error and plausible parameters. Until then, the ramp alone
  governs.
- **Cut fast, raise slowly.** Every cut applies immediately. A thermal cap rises again
  by at most `raise_w_per_s` (10 W/s); a budget share grows by at most
  `budget_raise_w_per_s`.
- **Sensor failure means base.** A failed or implausible reading (temperature outside
  0 to 125 °C, power outside 0 to 2000 W, and so on) puts that GPU at `base_w` until
  readings return, and it then recovers at the slow rate.
- **Every exit leaves a known limit.** On a stop, an error, a panic or a failed limit
  write, every GPU is set to `fallback_w` (default `base_w`), both from inside the
  process (Drop) and again by `ExecStopPost=gpu-powerd --restore`. The unit runs with
  `Restart=always` and a systemd watchdog. A lock file keeps it to a single instance.
- **Light on the driver.** NVML is opened once in-process (`dlopen` of
  `libnvidia-ml.so.1`); `nvidia-smi` is never spawned. The per-period reads
  (temperature, power, utilization, enforced limit, clock event reasons) cost about
  10 µs per GPU on an RTX PRO 6000 with driver 595. Fan speed goes to the GPU's
  firmware (1 to 7 ms per fan), so it is read every 10 s. Throttle counters and the
  T.Limit margin are never read. A limit is written only when its 5 W-quantised value
  changes, and rewritten every 30 s in case another tool changed it.

## How the budget is split

Every GPU first gets `base_w`. Demand then comes from load:

- An idle GPU (low utilization, not power-limited) wants `base_w`.
- A power-limited GPU (drawing at least 95% of its enforced limit, or reporting a
  power-cap clock event while busy) wants its full thermal cap.
- Any other busy GPU wants its draw plus `demand_headroom_w`.

The budget above the bases is water-filled up to each GPU's demand (max-min fair:
equal shares, and a GPU that wants less leaves the rest to the others). Whatever is
left over is filled up to each GPU's thermal cap, so an idle card holds some spare
power and can start work without waiting. Demand never exceeds a GPU's thermal cap,
so when a hot card is capped, the budget it cannot use goes to the cooler card,
within that card's own thermal limit.

## The thermal model

Each GPU is modelled as first order with self and cross coupling:

    τ_i · dT_i/dt = T_amb + R_ii·P_i + Σ_j R_ij·P_j − T_i

so `T_i → T_amb + R_ii·P_i + R_ij·P_j` with time constant `τ_i`. Readings are grouped
into 5 s windows. The model is fitted online by recursive least squares, with a
forgetting factor that applies only while the inputs move, so long idle periods do
not erase what was learned under load. Once confident, the model sets the limit at the
power whose predicted steady state is `throttle_start_c − model_margin_c`, with the
other GPUs at their current draw.

The model is saved to `/var/lib/gpu-powerd/model.json` every five minutes and on exit,
and is reloaded at start if the GPU set (by UUID) is unchanged.
`gpu-powerd model` shows R, τ, T_amb, the fit error, the confidence and the sample
count for each GPU.

In simulation (`tests/sim.rs`, a two-GPU plant where GPU 1 has poor airflow), four
hours of mixed load fit R within 5% and τ within 7%. On repeated 15-minute
full-load bursts, the ramp alone lets GPU 1 settle 5.2 °C above `throttle_start_c`
(about 11,900 °C·s above it); with the learned model, it stays at or below it.

## Install (Homebrew)

```sh
brew install tpurtell/local-ai/gpu-powerd
sudo "$(brew --prefix)/bin/gpu-powerd-setup"
```

gpu-powerd runs as root, so it must not execute from the Homebrew prefix, which your
user can write. The formula builds a root bundle (patched to the system loader, with
no embedded library paths, packed as a tarball). `gpu-powerd-setup` verifies the
bundle and installs root-owned copies:

- `/usr/local/sbin/gpu-powerd`
- `/etc/systemd/system/gpu-powerd.service`
- `/etc/gpu-powerd/config.toml` (written from the example only if absent)

It refuses to run on a host without an NVIDIA driver and GPUs. If
`nvidia-power-limit.service` is enabled, setup disables it and records that in
`/var/lib/gpu-powerd/replaced`. Run setup again after `brew upgrade gpu-powerd`.

```sh
sudo "$(brew --prefix)/bin/gpu-powerd-setup" --uninstall   # stop, remove, re-enable what it replaced
"$(brew --prefix)/bin/gpu-powerd-setup" --check            # verify the bundle (no root)
```

## Use

```sh
gpu-powerd --check            # read-only, no root: readings, board ranges, the plan, the model
gpu-powerd status             # the running daemon's limits and reasons (--json for the state file)
gpu-powerd model              # the learned model per GPU
sudo /usr/local/sbin/gpu-powerd set system_budget_w=1000 gpu.E1:00.0.max_w=450
sudo systemctl reload gpu-powerd   # after editing /etc/gpu-powerd/config.toml by hand
```

`set` edits the config in place (comments are kept), validates the result against the
GPUs present, and sends SIGHUP to the daemon. `KEY=` removes a setting. Run it as
root from `/usr/local/sbin`, never `sudo` the Homebrew copy. A reload that fails
validation is rejected, and the running config stays in force; `status` shows the
error.

## Configuration

See [`config.example.toml`](config.example.toml). The main keys:

| key | default | meaning |
|---|---|---|
| `system_budget_w` | unset (sum of `max_w`) | total for all GPUs |
| `throttle_start_c` | 75 | no thermal capping below this; the model's target |
| `hard_c` | 85 | at or above: straight to `base_w` |
| `base_w` / `max_w` | board min / max | per-GPU floor and ceiling (top level or `[gpu."SEL"]`) |
| `fallback_w` | `base_w` | the limit left on exit, crash or failed write |
| `raise_w_per_s` | 10 | how fast a thermal cap rises again |
| `use_model` | true | let a confident model lower limits |

GPUs are selected by NVML index (`[gpu.0]`), UUID (`[gpu."GPU-…"]`) or PCI bus id
(`[gpu."E1:00.0"]`).

## State file

`/run/gpu-powerd/state.json` is rewritten every period. It holds each GPU's applied
limit, its bound (`max`, `budget`, `ramp`, `model`, `recovering`, `hard`, `sensor`), a
readable reason, the readings, the smoothed values, the demand and share, and the
model summary. Other tools (such as a benchmark's throttle report) can read it. On
exit, `running` becomes false.

## Build and test

```sh
cargo test --release      # control law and learning against a simulated two-GPU plant
cargo run --release -- --check
cargo run --release --example nvml_timing   # read-only NVML call costs on this host
```

No runtime dependencies: one Rust binary, linked only against glibc. NVML is
`dlopen`ed at runtime, so there is no link-time dependency on the driver.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
