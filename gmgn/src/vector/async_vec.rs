//! Parallel vectorized environment backed by [rayon].
//!
//! Mirrors [Gymnasium `AsyncVectorEnv`](https://gymnasium.farama.org/api/vector/#gymnasium.vector.AsyncVectorEnv)
//! using rayon's work-stealing thread pool instead of per-env OS threads.
//!
//! The core operation — stepping N environments with N actions — is an
//! embarrassingly parallel map, which is exactly what rayon excels at.

use rayon::prelude::*;

use crate::env::{Env, RenderFrame};
use crate::error::{Error, Result};
use crate::vector::{AutoresetMode, VecResetResult, VecStepResult, VectorEnv};

/// Runs multiple copies of an environment **in parallel** using rayon's
/// work-stealing thread pool.
///
/// Structurally identical to [`SyncVectorEnv`](super::SyncVectorEnv) but
/// dispatches `step` / `reset` / `render` across rayon worker threads,
/// giving a genuine wall-clock speed-up for CPU-bound environments.
///
/// # Type Parameters
///
/// - `E` — The concrete [`Env`] type. Must be `Send` so it can be borrowed
///   across rayon threads.
///
/// # Examples
///
/// ```rust,ignore
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::vector::{AsyncVectorEnv, VectorEnv};
///
/// let envs: Vec<CartPoleEnv> = (0..4)
///     .map(|_| CartPoleEnv::new(CartPoleConfig::default()).unwrap())
///     .collect();
/// let mut vec_env = AsyncVectorEnv::new(envs).unwrap();
/// let reset = vec_env.reset(Some(42)).unwrap();
/// assert_eq!(reset.obs.len(), 4);
/// ```
#[derive(Debug)]
pub struct AsyncVectorEnv<E: Env> {
    envs: Vec<E>,
    autoreset_mode: AutoresetMode,
    needs_reset: Vec<bool>,
}

impl<E: Env> AsyncVectorEnv<E> {
    /// Create a new parallel vector environment from pre-built sub-envs.
    ///
    /// # Errors
    ///
    /// Returns an error if `envs` is empty.
    pub fn new(envs: Vec<E>) -> Result<Self> {
        if envs.is_empty() {
            return Err(Error::InvalidSpace {
                reason: "AsyncVectorEnv requires at least one sub-environment".to_owned(),
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

impl<E> VectorEnv for AsyncVectorEnv<E>
where
    E: Env + Send,
    E::Obs: Send,
    E::Act: Sync,
{
    type Obs = E::Obs;
    type Act = E::Act;

    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<VecResetResult<E::Obs>> {
        let results: Vec<Result<_>> = self
            .envs
            .par_iter_mut()
            .enumerate()
            .map(|(i, env)| {
                let s = seed.map(|s| s + i as u64);
                env.reset(s)
            })
            .collect();

        let mut obs = Vec::with_capacity(results.len());
        let mut infos = Vec::with_capacity(results.len());
        for r in results {
            let r = r?;
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

        let autoreset_mode = self.autoreset_mode;

        // Parallel step (with inline autoreset handling).
        let results: Vec<Result<_>> = self
            .envs
            .par_iter_mut()
            .zip(actions.par_iter())
            .zip(self.needs_reset.par_iter_mut())
            .map(|((env, action), needs_reset)| {
                match autoreset_mode {
                    AutoresetMode::NextStep => {
                        if *needs_reset {
                            let reset = env.reset(None)?;
                            *needs_reset = false;
                            return Ok((reset.obs, 0.0, false, false, reset.info));
                        }
                    }
                    AutoresetMode::SameStep | AutoresetMode::Disabled => {}
                }

                let r = env.step(action)?;
                let done = r.terminated || r.truncated;

                if done && autoreset_mode == AutoresetMode::SameStep {
                    let mut info = r.info;
                    info.insert(
                        "_final_observation".to_owned(),
                        crate::env::InfoValue::Bool(true),
                    );
                    let reset = env.reset(None)?;
                    *needs_reset = false;
                    Ok((reset.obs, r.reward, r.terminated, r.truncated, info))
                } else {
                    *needs_reset = done;
                    Ok((r.obs, r.reward, r.terminated, r.truncated, r.info))
                }
            })
            .collect();

        let n = results.len();
        let mut obs = Vec::with_capacity(n);
        let mut rewards = Vec::with_capacity(n);
        let mut terminated = Vec::with_capacity(n);
        let mut truncated = Vec::with_capacity(n);
        let mut infos = Vec::with_capacity(n);

        for r in results {
            let (o, reward, term, trunc, info) = r?;
            obs.push(o);
            rewards.push(reward);
            terminated.push(term);
            truncated.push(trunc);
            infos.push(info);
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
        self.envs.par_iter_mut().map(Env::render).collect()
    }

    fn close(&mut self) {
        self.envs.par_iter_mut().for_each(Env::close);
    }
}

/// CartPoleEnv contains raw window handles when the `render` feature is
/// enabled, making it `!Send`. Tests are gated accordingly.
#[cfg(test)]
#[cfg(not(feature = "render"))]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};

    fn make_vec(n: usize) -> AsyncVectorEnv<CartPoleEnv> {
        let envs: Vec<_> = (0..n)
            .map(|_| CartPoleEnv::new(CartPoleConfig::default()).unwrap())
            .collect();
        AsyncVectorEnv::new(envs).unwrap()
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
        let actions = vec![0_i64, 1]; // only 2
        assert!(v.step(&actions).is_err());
    }

    #[test]
    fn empty_envs_errors() {
        let result = AsyncVectorEnv::<CartPoleEnv>::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn autoreset_on_next_step() {
        let mut v = make_vec(1);
        v.reset(Some(0)).unwrap();

        let mut done = false;
        for _ in 0..500 {
            let r = v.step(&[1]).unwrap();
            if r.terminated[0] {
                done = true;
                break;
            }
        }
        assert!(done, "should terminate within 500 steps");

        // Next step should autoreset and succeed.
        let r = v.step(&[0]).unwrap();
        assert_eq!(r.obs.len(), 1);
    }

    #[test]
    fn num_envs_correct() {
        let v = make_vec(5);
        assert_eq!(v.num_envs(), 5);
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
    fn render_returns_frames() {
        let mut v = make_vec(2);
        v.reset(Some(0)).unwrap();
        let frames = v.render().unwrap();
        assert_eq!(frames.len(), 2);
    }
}
