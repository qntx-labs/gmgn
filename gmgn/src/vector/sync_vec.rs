//! Synchronous vectorized environment.
//!
//! Mirrors [Gymnasium `SyncVectorEnv`](https://gymnasium.farama.org/api/vector/#gymnasium.vector.SyncVectorEnv).

use crate::env::{Env, RenderFrame};
use crate::error::{Error, Result};
use crate::vector::{AutoresetMode, VecResetResult, VecStepResult, VectorEnv};

/// Runs multiple copies of an environment sequentially in a single thread.
///
/// Each sub-environment is stepped one after another. When a sub-environment
/// terminates or is truncated and [`AutoresetMode::NextStep`] is active, it is
/// automatically reset on the **next** call to [`step`](SyncVectorEnv::step).
///
/// # Type Parameters
///
/// - `E` — The concrete [`Env`] type for each sub-environment.
///
/// # Examples
///
/// ```rust
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::vector::{SyncVectorEnv, VectorEnv};
///
/// let envs: Vec<CartPoleEnv> = (0..4)
///     .map(|_| CartPoleEnv::new(CartPoleConfig::default()).unwrap())
///     .collect();
/// let mut vec_env = SyncVectorEnv::new(envs).unwrap();
/// let reset = vec_env.reset(Some(42)).unwrap();
/// assert_eq!(reset.obs.len(), 4);
/// ```
#[derive(Debug)]
pub struct SyncVectorEnv<E: Env> {
    envs: Vec<E>,
    autoreset_mode: AutoresetMode,
    /// Tracks which sub-envs need an autoreset on the next step.
    needs_reset: Vec<bool>,
}

impl<E: Env> SyncVectorEnv<E> {
    /// Create a new synchronous vector environment from a list of sub-envs.
    ///
    /// # Errors
    ///
    /// Returns an error if `envs` is empty.
    pub fn new(envs: Vec<E>) -> Result<Self> {
        if envs.is_empty() {
            return Err(Error::InvalidSpace {
                reason: "SyncVectorEnv requires at least one sub-environment".to_owned(),
            });
        }
        let n = envs.len();
        Ok(Self {
            envs,
            autoreset_mode: AutoresetMode::NextStep,
            needs_reset: vec![false; n],
        })
    }

    /// Create with a specific autoreset mode.
    ///
    /// # Errors
    ///
    /// Returns an error if `envs` is empty.
    pub fn with_autoreset(envs: Vec<E>, mode: AutoresetMode) -> Result<Self> {
        let mut v = Self::new(envs)?;
        v.autoreset_mode = mode;
        Ok(v)
    }

    /// The number of sub-environments.
    #[must_use]
    pub const fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// The autoreset mode.
    #[must_use]
    pub const fn autoreset_mode(&self) -> AutoresetMode {
        self.autoreset_mode
    }

    /// Borrow a single sub-environment by index.
    #[must_use]
    pub fn get_env(&self, index: usize) -> Option<&E> {
        self.envs.get(index)
    }

    /// Mutably borrow a single sub-environment by index.
    #[must_use]
    pub fn get_env_mut(&mut self, index: usize) -> Option<&mut E> {
        self.envs.get_mut(index)
    }
}

impl<E: Env> VectorEnv for SyncVectorEnv<E> {
    type Obs = E::Obs;
    type Act = E::Act;

    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<VecResetResult<E::Obs>> {
        let mut obs = Vec::with_capacity(self.envs.len());
        let mut infos = Vec::with_capacity(self.envs.len());

        for (i, env) in self.envs.iter_mut().enumerate() {
            let s = seed.map(|s| s + i as u64);
            let r = env.reset(s)?;
            obs.push(r.obs);
            infos.push(r.info);
        }

        self.needs_reset.fill(false);
        Ok(VecResetResult { obs, infos })
    }

    fn step(&mut self, actions: &[E::Act]) -> Result<VecStepResult<E::Obs>> {
        if actions.len() != self.envs.len() {
            return Err(Error::InvalidAction {
                reason: format!(
                    "expected {} actions, got {}",
                    self.envs.len(),
                    actions.len()
                ),
            });
        }

        let n = self.envs.len();
        let mut obs = Vec::with_capacity(n);
        let mut rewards = Vec::with_capacity(n);
        let mut terminated = Vec::with_capacity(n);
        let mut truncated = Vec::with_capacity(n);
        let mut infos = Vec::with_capacity(n);

        for (i, (env, action)) in self.envs.iter_mut().zip(actions.iter()).enumerate() {
            if self.needs_reset[i] && self.autoreset_mode == AutoresetMode::NextStep {
                env.reset(None)?;
            }

            let r = env.step(action)?;

            let done = r.terminated || r.truncated;
            self.needs_reset[i] = done;

            obs.push(r.obs);
            rewards.push(r.reward);
            terminated.push(r.terminated);
            truncated.push(r.truncated);
            infos.push(r.info);
        }

        Ok(VecStepResult {
            obs,
            rewards,
            terminated,
            truncated,
            infos,
        })
    }

    fn render(&mut self) -> Result<Vec<RenderFrame>> {
        self.envs.iter_mut().map(Env::render).collect()
    }

    fn close(&mut self) {
        for env in &mut self.envs {
            env.close();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};

    fn make_vec(n: usize) -> SyncVectorEnv<CartPoleEnv> {
        let envs: Vec<_> = (0..n)
            .map(|_| CartPoleEnv::new(CartPoleConfig::default()).unwrap())
            .collect();
        SyncVectorEnv::new(envs).unwrap()
    }

    #[test]
    fn reset_returns_n_observations() {
        let mut v = make_vec(4);
        let r = v.reset(Some(42)).unwrap();
        assert_eq!(r.obs.len(), 4);
        assert_eq!(r.infos.len(), 4);
    }

    #[test]
    fn step_returns_n_results() {
        let mut v = make_vec(3);
        v.reset(Some(0)).unwrap();
        let actions = vec![0_i64, 1, 0];
        let r = v.step(&actions).unwrap();
        assert_eq!(r.obs.len(), 3);
        assert_eq!(r.rewards.len(), 3);
        assert_eq!(r.terminated.len(), 3);
        assert_eq!(r.truncated.len(), 3);
    }

    #[test]
    fn step_wrong_action_count_errors() {
        let mut v = make_vec(3);
        v.reset(Some(0)).unwrap();
        let actions = vec![0_i64, 1]; // only 2 instead of 3
        assert!(v.step(&actions).is_err());
    }

    #[test]
    fn empty_envs_errors() {
        let result = SyncVectorEnv::<CartPoleEnv>::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn deterministic_with_seed() {
        let mut v1 = make_vec(2);
        let mut v2 = make_vec(2);

        let r1 = v1.reset(Some(99)).unwrap();
        let r2 = v2.reset(Some(99)).unwrap();
        assert_eq!(r1.obs, r2.obs);
    }

    #[test]
    fn autoreset_on_next_step() {
        let mut v = make_vec(1);
        v.reset(Some(0)).unwrap();

        // Step until termination.
        let mut done = false;
        for _ in 0..500 {
            let r = v.step(&[1]).unwrap();
            if r.terminated[0] {
                done = true;
                break;
            }
        }
        assert!(done, "should terminate within 500 steps");

        // Next step should autoreset and succeed (not error out).
        let r = v.step(&[0]).unwrap();
        assert_eq!(r.obs.len(), 1);
    }

    #[test]
    fn num_envs_correct() {
        let v = make_vec(5);
        assert_eq!(v.num_envs(), 5);
    }

    #[test]
    fn render_returns_frames() {
        let mut v = make_vec(2);
        v.reset(Some(0)).unwrap();
        let frames = v.render().unwrap();
        assert_eq!(frames.len(), 2);
    }
}
