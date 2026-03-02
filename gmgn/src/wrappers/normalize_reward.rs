//! Running-mean reward normalization wrapper.
//!
//! Mirrors [Gymnasium `NormalizeReward`](https://gymnasium.farama.org/api/wrappers/reward_wrappers/#gymnasium.wrappers.NormalizeReward).

use crate::env::{Env, ResetResult, StepResult};
use crate::error::Result;
use crate::macros::delegate_env;

/// Normalizes rewards using a running discounted return variance estimate.
///
/// Uses Welford's online algorithm to track the variance of discounted
/// returns, then divides each reward by `sqrt(var + epsilon)`.
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::NormalizeReward;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = NormalizeReward::new(env, 0.99, 1e-8);
/// ```
#[derive(Debug)]
pub struct NormalizeReward<E: Env> {
    env: E,
    /// Discount factor for return estimation.
    gamma: f64,
    /// Small constant for numerical stability.
    epsilon: f64,
    /// Running discounted return estimate.
    ret: f64,
    /// Running count.
    count: f64,
    /// Running mean of returns.
    mean: f64,
    /// Running variance numerator (sum of squared deviations).
    var: f64,
}

impl<E: Env> NormalizeReward<E> {
    /// Wrap `env` with running reward normalization.
    ///
    /// - `gamma` — discount factor (e.g. `0.99`).
    /// - `epsilon` — numerical stability constant (e.g. `1e-8`).
    #[must_use]
    pub const fn new(env: E, gamma: f64, epsilon: f64) -> Self {
        Self {
            env,
            gamma,
            epsilon,
            ret: 0.0,
            count: 0.0,
            mean: 0.0,
            var: 1.0,
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

    /// Update running statistics and return the normalized reward.
    fn normalize_reward(&mut self, reward: f64) -> f64 {
        self.ret = self.ret.mul_add(self.gamma, reward);

        // Welford's online update with the current return.
        self.count += 1.0;
        let delta = self.ret - self.mean;
        self.mean += delta / self.count;
        let delta2 = self.ret - self.mean;
        self.var += delta * delta2;

        let std = (self.var / self.count + self.epsilon).sqrt();
        reward / std
    }
}

impl<E: Env> Env for NormalizeReward<E> {
    type Obs = E::Obs;
    type Act = E::Act;
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let mut result = self.env.step(action)?;
        result.reward = self.normalize_reward(result.reward);

        // Reset the return accumulator on episode end.
        if result.terminated || result.truncated {
            self.ret = 0.0;
        }

        Ok(result)
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
        self.ret = 0.0;
        self.env.reset(seed)
    }

    delegate_env!(env);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};

    #[test]
    fn rewards_are_finite() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = NormalizeReward::new(env, 0.99, 1e-8);
        env.reset(Some(42)).unwrap();

        for _ in 0..100 {
            let r = env.step(&0).unwrap();
            assert!(r.reward.is_finite(), "reward should be finite");
            if r.terminated {
                env.reset(None).unwrap();
            }
        }
    }

    #[test]
    fn reset_clears_return() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = NormalizeReward::new(env, 0.99, 1e-8);
        env.reset(Some(0)).unwrap();
        env.step(&0).unwrap();
        env.reset(Some(1)).unwrap();
        assert!((env.ret - 0.0).abs() < f64::EPSILON);
    }
}
