//! gpu-powerd: a thermal- and budget-aware NVIDIA power-limit daemon.
//!
//! [`control`] is the control law, [`model`] the learned thermal model, [`daemon`]
//! drives both over a [`backend::Backend`] (NVML on hosts, [`sim`] in tests).
//! [`stream`] publishes samples locally; `gpu-powerd-ws` relays them over [`websocket`].
pub mod backend;
pub mod config;
pub mod control;
pub mod daemon;
pub mod model;
pub mod nvml;
pub mod sim;
pub mod stream;
pub mod websocket;
