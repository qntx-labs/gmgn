//! Truncates episodes after a fixed number of steps.
//!
//! Mirrors [Gymnasium `TimeLimit`](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.TimeLimit).

use crate::env::{Env, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::Result;

/// Limits the number of steps per episode, setting `truncated = true` when the
/// budget is exhausted.
///
/// This wrapper does **not** reset the environment automatically; it only
/// signals truncation. Use an `Autoreset` layer if auto-reset is desired.
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::TimeLimit;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = TimeLimit::new(env, 500);
/// let _reset = env.reset(Some(42)).unwrap();
/// ```
#[derive(Debug)]
pub struct TimeLimit<E: Env> {
    /// The wrapped environment.
    env: E,
    /// Maximum number of steps per episode before truncation.
    max_episode_steps: u64,
    /// Steps elapsed in the current episode (`None` before first reset).
    elapsed_steps: Option<u64>,
}

impl<E: Env> TimeLimit<E> {
    /// Wrap `env` with a step budget of `max_episode_steps`.
    ///
    /// # Panics
    ///
    /// Panics if `max_episode_steps` is zero.
    #[must_use]
    pub fn new(env: E, max_episode_steps: u64) -> Self {
        assert!(max_episode_steps > 0, "max_episode_steps must be positive");
        Self {
            env,
            max_episode_steps,
            elapsed_steps: None,
        }
    }

    /// The maximum steps allowed per episode.
    #[must_use]
    pub const fn max_episode_steps(&self) -> u64 {
        self.max_episode_steps
    }

    /// Steps elapsed in the current episode, or `None` if not yet reset.
    #[must_use]
    pub const fn elapsed_steps(&self) -> Option<u64> {
        self.elapsed_steps
    }

    /// Borrow the inner environment.
    #[must_use]
    pub const fn inner(&self) -> &E {
        &self.env
    }

    /// Mutably borrow the inner environment.
    #[must_use]
    pub const fn inner_mut(&mut self) -> &mut E {
        &mut self.env
    }

    /// Unwrap and return the inner environment.
    #[must_use]
    pub fn into_inner(self) -> E {
        self.env
    }
}

impl<E: Env> Env for TimeLimit<E> {
    type Obs = E::Obs;
    type Act = E::Act;
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let mut result = self.env.step(action)?;

        // Increment elapsed counter (starts at 0 after reset).
        let elapsed = self
            .elapsed_steps
            .as_mut()
            .expect("environment must be reset before step");
        *elapsed += 1;

        if *elapsed >= self.max_episode_steps {
            result.truncated = true;
        }

        Ok(result)
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
        self.elapsed_steps = Some(0);
        self.env.reset(seed)
    }

    fn render(&mut self) -> Result<RenderFrame> {
        self.env.render()
    }

    fn close(&mut self) {
        self.env.close();
    }

    fn observation_space(&self) -> &Self::ObsSpace {
        self.env.observation_space()
    }

    fn action_space(&self) -> &Self::ActSpace {
        self.env.action_space()
    }

    fn render_mode(&self) -> &RenderMode {
        self.env.render_mode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};
    use crate::space::Space;

    #[test]
    fn truncates_at_limit() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = TimeLimit::new(env, 3);
        env.reset(Some(42)).unwrap();

        let r1 = env.step(&0).unwrap();
        assert!(!r1.truncated);
        assert_eq!(env.elapsed_steps(), Some(1));

        let r2 = env.step(&1).unwrap();
        assert!(!r2.truncated);

        let r3 = env.step(&0).unwrap();
        assert!(r3.truncated, "should truncate at step 3");
        assert_eq!(env.elapsed_steps(), Some(3));
    }

    #[test]
    fn reset_clears_counter() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = TimeLimit::new(env, 10);

        env.reset(Some(0)).unwrap();
        env.step(&0).unwrap();
        assert_eq!(env.elapsed_steps(), Some(1));

        env.reset(Some(1)).unwrap();
        assert_eq!(env.elapsed_steps(), Some(0));
    }

    #[test]
    fn delegates_spaces() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let wrapped = TimeLimit::new(env, 500);
        assert_eq!(wrapped.observation_space().shape(), &[4]);
        assert_eq!(wrapped.action_space().n, 2);
    }
}
