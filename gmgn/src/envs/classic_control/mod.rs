//! Classic control environments.
//!
//! These are simple continuous-state environments commonly used as
//! benchmarks for reinforcement learning algorithms.
//!
//! Mirrors [Gymnasium classic control](https://gymnasium.farama.org/environments/classic_control/).

mod acrobot;
mod cartpole;
mod mountain_car;
mod pendulum;

pub use acrobot::{AcrobotConfig, AcrobotEnv};
pub use cartpole::{CartPoleConfig, CartPoleEnv};
pub use mountain_car::{MountainCarConfig, MountainCarEnv};
pub use pendulum::{PendulumConfig, PendulumEnv};
