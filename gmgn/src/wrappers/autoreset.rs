//! Automatically resets the environment when an episode ends.
//!
//! Mirrors [Gymnasium `Autoreset`](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.Autoreset).

use crate::env::{Env, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::Result;

/// Automatically resets the environment when `terminated` or `truncated`
/// is `true`, returning the new episode's initial observation.
///
/// When an episode ends, the wrapper calls [`reset`](Env::reset) internally
/// and replaces the terminal observation with the fresh initial observation.
/// The terminal observation is stored in `info` under the key
/// `"terminal_observation"` (as a boolean flag indicating it was replaced).
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::Autoreset;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = Autoreset::new(env);
/// env.reset(Some(42)).unwrap();
/// // After the episode ends, the next observation is from a fresh reset.
/// ```
#[derive(Debug)]
pub struct Autoreset<E: Env> {
    env: E,
}

impl<E: Env> Autoreset<E> {
    /// Wrap `env` with automatic episode resets.
    #[must_use]
    pub const fn new(env: E) -> Self {
        Self { env }
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

impl<E: Env> Env for Autoreset<E> {
    type Obs = E::Obs;
    type Act = E::Act;
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let result = self.env.step(action)?;

        if result.terminated || result.truncated {
            // Auto-reset and return the new episode's initial observation.
            let reset = self.env.reset(None)?;
            let mut info = result.info;
            info.insert("autoreset".to_owned(), crate::env::InfoValue::Bool(true));
            return Ok(StepResult {
                obs: reset.obs,
                reward: result.reward,
                terminated: result.terminated,
                truncated: result.truncated,
                info,
            });
        }

        Ok(result)
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
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

    #[test]
    fn autoreset_on_termination() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = Autoreset::new(env);
        env.reset(Some(0)).unwrap();

        // Step until termination.
        let mut terminated = false;
        for _ in 0..500 {
            let r = env.step(&1).unwrap();
            if r.terminated || r.truncated {
                terminated = true;
                // The observation should be from the fresh reset.
                assert_eq!(r.obs.len(), 4);
                assert!(r.info.contains_key("autoreset"));
                break;
            }
        }
        assert!(terminated, "should terminate within 500 steps");

        // Should be able to continue stepping without manual reset.
        let r = env.step(&0).unwrap();
        assert_eq!(r.obs.len(), 4);
    }
}
