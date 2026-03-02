//! Vectorized environments for batched parallel execution.
//!
//! Running multiple copies of the same environment simultaneously enables
//! efficient data collection for reinforcement learning training.
//!
//! Mirrors [Gymnasium vector](https://gymnasium.farama.org/api/vector/)
//! adapted to Rust's ownership model.

mod async_vec;
mod sync_vec;
mod wrapper;

pub use async_vec::AsyncVectorEnv;
pub use sync_vec::SyncVectorEnv;
pub use wrapper::VecRecordEpisodeStatistics;

use crate::env::{Info, RenderFrame};
use crate::error::Result;

/// Batched results from [`SyncVectorEnv::step`].
#[derive(Debug, Clone)]
pub struct VecStepResult<O> {
    /// One observation per sub-environment.
    pub obs: Vec<O>,
    /// One scalar reward per sub-environment.
    pub rewards: Vec<f64>,
    /// Per-environment MDP termination flags.
    pub terminated: Vec<bool>,
    /// Per-environment truncation flags.
    pub truncated: Vec<bool>,
    /// Per-environment auxiliary info dicts.
    pub infos: Vec<Info>,
}

/// Batched results from [`SyncVectorEnv::reset`].
#[derive(Debug, Clone)]
pub struct VecResetResult<O> {
    /// One observation per sub-environment.
    pub obs: Vec<O>,
    /// Per-environment auxiliary info dicts.
    pub infos: Vec<Info>,
}

/// Autoreset behaviour when a sub-environment episode ends.
///
/// Mirrors [Gymnasium `AutoresetMode`](https://gymnasium.farama.org/api/vector/#gymnasium.vector.AutoresetMode).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AutoresetMode {
    /// Reset on the **next** call to `step` (default, matches Gymnasium).
    ///
    /// When a sub-env terminates, the *next* `step` call resets it and
    /// returns the reset observation with zero reward instead of stepping.
    #[default]
    NextStep,
    /// Reset in the **same** `step` call that caused termination.
    ///
    /// The final observation and info are stored under `"final_obs"` and
    /// `"final_info"` in the info dict, and the returned observation is
    /// the fresh post-reset observation.
    SameStep,
    /// Disabled — the caller is responsible for resetting.
    Disabled,
}

/// Common interface for vectorized environments.
///
/// Abstracts over [`SyncVectorEnv`] (and future `AsyncVectorEnv`),
/// mirroring [Gymnasium `VectorEnv`](https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv).
pub trait VectorEnv {
    /// The observation type of each sub-environment.
    type Obs;
    /// The action type of each sub-environment.
    type Act;

    /// The number of sub-environments.
    fn num_envs(&self) -> usize;

    /// Reset all sub-environments.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during reset.
    fn reset(&mut self, seed: Option<u64>) -> Result<VecResetResult<Self::Obs>>;

    /// Step all sub-environments with one action per env.
    ///
    /// # Errors
    ///
    /// Returns an error if `actions.len() != num_envs()` or any sub-env fails.
    fn step(&mut self, actions: &[Self::Act]) -> Result<VecStepResult<Self::Obs>>;

    /// Render all sub-environments, returning one frame per env.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during rendering.
    fn render(&mut self) -> Result<Vec<RenderFrame>>;

    /// Close all sub-environments.
    fn close(&mut self);
}
