//! # gmgn — Rust Reinforcement Learning Gymnasium
//!
//! A pure-Rust library for reinforcement learning environments, inspired by
//! [Gymnasium](https://gymnasium.farama.org/). Designed for zero-copy
//! integration with the [burn](https://burn.dev/) deep learning framework.
//!
//! ## Quick Start
//!
//! ```rust
//! use gmgn::prelude::*;
//! use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
//!
//! let mut env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
//! let reset = env.reset(Some(42)).unwrap();
//!
//! for _ in 0..200 {
//!     let action = env.action_space().sample(&mut gmgn::rng::create_rng(None));
//!     let step = env.step(&action).unwrap();
//!     if step.terminated {
//!         break;
//!     }
//! }
//! ```

pub mod env;
pub mod envs;
pub mod error;
pub mod prelude;
#[cfg(feature = "render")]
pub mod render;
pub mod rng;
pub mod space;
pub mod wrapper;
