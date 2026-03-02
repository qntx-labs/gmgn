//! Applies an arbitrary function to the reward.
//!
//! Mirrors [Gymnasium `TransformReward`](https://gymnasium.farama.org/api/wrappers/reward_wrappers/#gymnasium.wrappers.TransformReward).

use crate::env::{Env, StepResult};
use crate::error::Result;
use crate::macros::delegate_env;

/// Applies a user-supplied function `f(reward) -> reward` after each step.
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::TransformReward;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = TransformReward::new(env, |r| r * 0.1); // scale reward
/// let _reset = env.reset(Some(42)).unwrap();
/// ```
pub struct TransformReward<E: Env, F: Fn(f64) -> f64> {
    env: E,
    f: F,
}

impl<E: Env + std::fmt::Debug, F: Fn(f64) -> f64> std::fmt::Debug for TransformReward<E, F> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("TransformReward")
            .field("env", &self.env)
            .finish_non_exhaustive()
    }
}

impl<E: Env, F: Fn(f64) -> f64> TransformReward<E, F> {
    /// Wrap `env` so that rewards are transformed by `f`.
    #[must_use]
    pub const fn new(env: E, f: F) -> Self {
        Self { env, f }
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

impl<E: Env, F: Fn(f64) -> f64> Env for TransformReward<E, F> {
    type Obs = E::Obs;
    type Act = E::Act;
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let mut result = self.env.step(action)?;
        result.reward = (self.f)(result.reward);
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
    fn scales_reward() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = TransformReward::new(env, |r| r * 0.1);
        env.reset(Some(42)).unwrap();

        // CartPole reward is 1.0; scaled should be 0.1.
        let r = env.step(&0).unwrap();
        assert!((r.reward - 0.1).abs() < 1e-9);
    }

    #[test]
    fn negates_reward() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = TransformReward::new(env, |r| -r);
        env.reset(Some(42)).unwrap();

        let r = env.step(&0).unwrap();
        assert!((r.reward + 1.0).abs() < 1e-9);
    }
}
