//! Box2D-based environments requiring the `box2d` feature.
//!
//! These environments use the [`box2d-rs`](https://crates.io/crates/box2d-rs)
//! pure-Rust port of the `Box2D` physics engine.

mod bipedal_walker;
mod lunar_lander;

pub use bipedal_walker::{BipedalWalkerConfig, BipedalWalkerEnv};
pub use lunar_lander::{LunarLanderConfig, LunarLanderEnv};
