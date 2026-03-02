//! Classic control environments.
//!
//! These are simple continuous-state environments commonly used as
//! benchmarks for reinforcement learning algorithms.

mod cartpole;

pub use cartpole::{CartPoleConfig, CartPoleEnv};
