//! Clips rewards to a bounded range.
//!
//! Mirrors [Gymnasium `ClipReward`](https://gymnasium.farama.org/api/wrappers/reward_wrappers/#gymnasium.wrappers.ClipReward).

use crate::env::{Env, StepResult};
use crate::error::Result;
use crate::macros::delegate_env;

/// Clips the reward returned by [`step`](Env::step) to `[min_reward, max_reward]`.
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::ClipReward;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = ClipReward::new(env, -1.0, 1.0);
/// let _reset = env.reset(Some(42)).unwrap();
/// ```
#[derive(Debug)]
pub struct ClipReward<E: Env> {
    env: E,
    min_reward: f64,
    max_reward: f64,
}

impl<E: Env> ClipReward<E> {
    /// Wrap `env` so rewards are clipped to `[min_reward, max_reward]`.
    ///
    /// # Panics
    ///
    /// Panics if `min_reward > max_reward`.
    #[must_use]
    pub fn new(env: E, min_reward: f64, max_reward: f64) -> Self {
        assert!(
            min_reward <= max_reward,
            "min_reward ({min_reward}) > max_reward ({max_reward})"
        );
        Self {
            env,
            min_reward,
            max_reward,
        }
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

impl<E: Env> Env for ClipReward<E> {
    type Obs = E::Obs;
    type Act = E::Act;
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let mut result = self.env.step(action)?;
        result.reward = result.reward.clamp(self.min_reward, self.max_reward);
        Ok(result)
    }

    delegate_env!(
        env,
        reset,
        render,
        close,
        render_mode,
        observation_space,
        action_space
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};

    #[test]
    fn clips_reward_to_range() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = ClipReward::new(env, 0.0, 0.5);
        env.reset(Some(42)).unwrap();

        // CartPole gives reward=1.0 per step; should be clipped to 0.5.
        let r = env.step(&0).unwrap();
        assert!(r.reward <= 0.5);
        assert!(r.reward >= 0.0);
    }

    #[test]
    #[should_panic(expected = "min_reward")]
    fn rejects_inverted_bounds() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let _ = ClipReward::new(env, 1.0, -1.0);
    }
}
