//! Returns only every `skip`-th frame, taking the element-wise max of the last two.
//!
//! Mirrors [Gymnasium `MaxAndSkipObservation`](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.MaxAndSkipObservation).

use crate::env::{Env, StepResult};
use crate::error::Result;
use crate::macros::delegate_env;

/// Skips `skip` frames and returns the element-wise maximum of the last two
/// observed frames. Rewards are summed over skipped frames.
///
/// This is a standard technique for Atari environments to reduce temporal
/// aliasing caused by sprite flickering.
///
/// Only applicable to environments whose `Obs` is `Vec<f32>`.
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::MaxAndSkipObservation;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = MaxAndSkipObservation::new(env, 4).unwrap();
/// ```
#[derive(Debug)]
pub struct MaxAndSkipObservation<E>
where
    E: Env<Obs = Vec<f32>>,
{
    env: E,
    skip: usize,
}

impl<E> MaxAndSkipObservation<E>
where
    E: Env<Obs = Vec<f32>>,
{
    /// Wrap `env` to skip `skip` frames, returning the max of the last two.
    ///
    /// # Errors
    ///
    /// Returns an error if `skip` is zero.
    pub fn new(env: E, skip: usize) -> Result<Self> {
        if skip == 0 {
            return Err(crate::error::Error::InvalidSpace {
                reason: "skip must be >= 1".to_owned(),
            });
        }
        Ok(Self { env, skip })
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

impl<E> Env for MaxAndSkipObservation<E>
where
    E: Env<Obs = Vec<f32>>,
{
    type Obs = Vec<f32>;
    type Act = E::Act;
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let mut total_reward = 0.0;
        let mut prev_obs: Option<Vec<f32>> = None;
        let mut result = self.env.step(action)?;

        for i in 0..self.skip {
            if i > 0 {
                result = self.env.step(action)?;
            }
            total_reward += result.reward;

            if result.terminated || result.truncated {
                // Keep current obs as the last frame.
                break;
            }

            // Retain the second-to-last observation.
            if i == self.skip.saturating_sub(2) {
                prev_obs = Some(result.obs.clone());
            }
        }

        // Element-wise max of last two frames.
        if let Some(prev) = prev_obs {
            let maxed: Vec<f32> = result
                .obs
                .iter()
                .zip(prev.iter())
                .map(|(&a, &b)| a.max(b))
                .collect();
            result.obs = maxed;
        }

        result.reward = total_reward;
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
    fn skip_accumulates_reward() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = MaxAndSkipObservation::new(env, 4).unwrap();
        env.reset(Some(42)).unwrap();
        let r = env.step(&0).unwrap();
        // CartPole gives 1.0 per step; 4 steps → 4.0 (if not terminated).
        assert!(r.reward >= 1.0);
        assert_eq!(r.obs.len(), 4);
    }

    #[test]
    fn skip_one_is_passthrough() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = MaxAndSkipObservation::new(env, 1).unwrap();
        env.reset(Some(42)).unwrap();
        let r = env.step(&0).unwrap();
        assert!((r.reward - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rejects_zero_skip() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        assert!(MaxAndSkipObservation::new(env, 0).is_err());
    }
}
