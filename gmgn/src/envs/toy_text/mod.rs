//! Toy text environments.
//!
//! Simple grid-world and tabular environments commonly used for teaching
//! reinforcement learning fundamentals.
//!
//! Mirrors [Gymnasium toy text](https://gymnasium.farama.org/environments/toy_text/).

mod cliff_walking;
mod frozen_lake;
mod taxi;

pub use cliff_walking::{CliffWalkingConfig, CliffWalkingEnv};
pub use frozen_lake::{FrozenLakeConfig, FrozenLakeEnv};
pub use taxi::{TaxiConfig, TaxiEnv};
