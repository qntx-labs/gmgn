//! Toy text environments.
//!
//! Simple grid-world and tabular environments commonly used for teaching
//! reinforcement learning fundamentals.
//!
//! Mirrors [Gymnasium toy text](https://gymnasium.farama.org/environments/toy_text/).

mod frozen_lake;

pub use frozen_lake::{FrozenLakeConfig, FrozenLakeEnv};
