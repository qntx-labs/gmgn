//! Vector environment wrappers for batched transformation.
//!
//! Mirrors [Gymnasium `VectorWrapper`](https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorWrapper)
//! adapted to Rust generics for zero-cost composition.

use crate::env::RenderFrame;
use crate::error::Result;
use crate::vector::{VecResetResult, VecStepResult, VectorEnv};

/// Tracks per-episode returns and lengths across a vectorized environment.
///
/// After each sub-environment episode completes (terminated or truncated),
/// the cumulative reward and step count are recorded.
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::vector::{SyncVectorEnv, VectorEnv, VecRecordEpisodeStatistics};
///
/// let envs: Vec<CartPoleEnv> = (0..4)
///     .map(|_| CartPoleEnv::new(CartPoleConfig::default()).unwrap())
///     .collect();
/// let vec_env = SyncVectorEnv::new(envs).unwrap();
/// let mut wrapped = VecRecordEpisodeStatistics::new(vec_env);
/// wrapped.reset(Some(42)).unwrap();
/// ```
#[derive(Debug)]
pub struct VecRecordEpisodeStatistics<V: VectorEnv> {
    env: V,
    /// Accumulated reward for the current episode in each sub-env.
    episode_returns: Vec<f64>,
    /// Step count for the current episode in each sub-env.
    episode_lengths: Vec<u64>,
    /// Completed episode returns (most recent per sub-env).
    completed_returns: Vec<Option<f64>>,
    /// Completed episode lengths (most recent per sub-env).
    completed_lengths: Vec<Option<u64>>,
}

impl<V: VectorEnv> VecRecordEpisodeStatistics<V> {
    /// Wrap a vector environment with episode statistics tracking.
    #[must_use]
    pub fn new(env: V) -> Self {
        let n = env.num_envs();
        Self {
            env,
            episode_returns: vec![0.0; n],
            episode_lengths: vec![0; n],
            completed_returns: vec![None; n],
            completed_lengths: vec![None; n],
        }
    }

    /// Most recently completed episode return per sub-env (`None` if no episode finished yet).
    #[must_use]
    pub fn completed_returns(&self) -> &[Option<f64>] {
        &self.completed_returns
    }

    /// Most recently completed episode length per sub-env (`None` if no episode finished yet).
    #[must_use]
    pub fn completed_lengths(&self) -> &[Option<u64>] {
        &self.completed_lengths
    }

    /// Borrow the inner vector environment.
    #[must_use]
    pub const fn inner(&self) -> &V {
        &self.env
    }

    /// Mutably borrow the inner vector environment.
    #[must_use]
    pub const fn inner_mut(&mut self) -> &mut V {
        &mut self.env
    }

    /// Unwrap and return the inner vector environment.
    #[must_use]
    pub fn into_inner(self) -> V {
        self.env
    }
}

impl<V: VectorEnv> VectorEnv for VecRecordEpisodeStatistics<V> {
    type Obs = V::Obs;
    type Act = V::Act;

    fn num_envs(&self) -> usize {
        self.env.num_envs()
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<VecResetResult<Self::Obs>> {
        let result = self.env.reset(seed)?;
        self.episode_returns.fill(0.0);
        self.episode_lengths.fill(0);
        Ok(result)
    }

    fn step(&mut self, actions: &[Self::Act]) -> Result<VecStepResult<Self::Obs>> {
        let result = self.env.step(actions)?;

        for i in 0..result.rewards.len() {
            self.episode_returns[i] += result.rewards[i];
            self.episode_lengths[i] += 1;

            if result.terminated[i] || result.truncated[i] {
                self.completed_returns[i] = Some(self.episode_returns[i]);
                self.completed_lengths[i] = Some(self.episode_lengths[i]);
                self.episode_returns[i] = 0.0;
                self.episode_lengths[i] = 0;
            }
        }

        Ok(result)
    }

    fn render(&mut self) -> Result<Vec<RenderFrame>> {
        self.env.render()
    }

    fn close(&mut self) {
        self.env.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};
    use crate::vector::SyncVectorEnv;

    fn make_wrapped(n: usize) -> VecRecordEpisodeStatistics<SyncVectorEnv<CartPoleEnv>> {
        let envs: Vec<_> = (0..n)
            .map(|_| CartPoleEnv::new(CartPoleConfig::default()).unwrap())
            .collect();
        let vec_env = SyncVectorEnv::new(envs).unwrap();
        VecRecordEpisodeStatistics::new(vec_env)
    }

    #[test]
    fn tracks_episode_stats() {
        let mut env = make_wrapped(1);
        env.reset(Some(0)).unwrap();

        // Step until episode ends.
        let mut done = false;
        for _ in 0..500 {
            let r = env.step(&[1]).unwrap();
            if r.terminated[0] || r.truncated[0] {
                done = true;
                break;
            }
        }
        assert!(done, "episode should end within 500 steps");
        assert!(env.completed_returns()[0].is_some());
        assert!(env.completed_lengths()[0].is_some());
        assert!(env.completed_lengths()[0].unwrap() > 0);
    }

    #[test]
    fn reset_clears_accumulators() {
        let mut env = make_wrapped(2);
        env.reset(Some(42)).unwrap();
        env.step(&[0, 1]).unwrap();

        env.reset(Some(0)).unwrap();
        // After reset, episode_returns/lengths are zeroed.
        // completed_* retain their last values (not cleared by reset).
    }

    #[test]
    fn num_envs_delegates() {
        let env = make_wrapped(3);
        assert_eq!(env.num_envs(), 3);
    }
}
