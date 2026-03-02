//! # gmgn — Rust Reinforcement Learning Gymnasium
//!
//! A pure-Rust library for reinforcement learning environments, inspired by
//! [Gymnasium](https://gymnasium.farama.org/). Designed for zero-cost
//! abstraction and idiomatic Rust ergonomics.
//!
//! ## Quick Start (static, generic)
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
//!
//! ## Quick Start (dynamic, registry)
//!
//! ```rust
//! use gmgn::registry::{self, DynValue};
//!
//! registry::register_builtins();
//! let mut env = registry::make("CartPole-v1").unwrap();
//! env.reset_dyn(Some(42)).unwrap();
//! let step = env.step_dyn(&DynValue::Discrete(1)).unwrap();
//! ```

pub mod env;
pub mod envs;
pub mod error;
pub mod prelude;
pub mod registry;
#[cfg(feature = "render")]
pub mod render;
pub mod rng;
pub mod space;
pub mod vector;
pub mod wrapper;
pub mod wrappers;
