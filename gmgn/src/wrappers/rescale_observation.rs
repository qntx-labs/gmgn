//! Rescales continuous observations from the environment's bounds to a new range.
//!
//! Mirrors [Gymnasium `RescaleObservation`](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.RescaleObservation).

use crate::env::{Env, ResetResult, StepResult};
use crate::error::{Error, Result};
use crate::macros::delegate_env;
use crate::space::BoundedSpace;

/// Linearly rescales each element of a continuous observation from the
/// environment's original bounds to `[min_obs, max_obs]`.
///
/// Only applicable to environments whose `Obs` is `Vec<f32>` and `ObsSpace`
/// is [`BoundedSpace`].
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::RescaleObservation;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = RescaleObservation::new(env, -1.0, 1.0).unwrap();
/// ```
#[derive(Debug)]
pub struct RescaleObservation<E>
where
    E: Env<Obs = Vec<f32>, ObsSpace = BoundedSpace>,
{
    env: E,
    min_obs: Vec<f32>,
    max_obs: Vec<f32>,
    new_space: BoundedSpace,
}

impl<E> RescaleObservation<E>
where
    E: Env<Obs = Vec<f32>, ObsSpace = BoundedSpace>,
{
    /// Wrap `env` to rescale observations into `[min_obs, max_obs]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `min_obs >= max_obs`.
    pub fn new(env: E, min_obs: f32, max_obs: f32) -> Result<Self> {
        if min_obs >= max_obs {
            return Err(Error::InvalidSpace {
                reason: format!("min_obs ({min_obs}) must be < max_obs ({max_obs})"),
            });
        }

        let inner = env.observation_space();
        let dim = inner.low.len();

        let low = vec![min_obs; dim];
        let high = vec![max_obs; dim];
        let new_space =
            BoundedSpace::new(low.clone(), high.clone()).map_err(|e| Error::InvalidSpace {
                reason: format!("failed to create rescaled observation space: {e}"),
            })?;

        Ok(Self {
            env,
            min_obs: low,
            max_obs: high,
            new_space,
        })
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

    /// Rescale a single observation from original bounds to new bounds.
    fn rescale(&self, obs: &[f32]) -> Vec<f32> {
        let inner = self.env.observation_space();
        obs.iter()
            .zip(inner.low.iter().zip(inner.high.iter()))
            .zip(self.min_obs.iter().zip(self.max_obs.iter()))
            .map(|((&v, (&old_lo, &old_hi)), (&new_lo, &new_hi))| {
                let range = old_hi - old_lo;
                if range.abs() < f32::EPSILON {
                    new_lo
                } else {
                    let t = (v - old_lo) / range;
                    t.mul_add(new_hi - new_lo, new_lo)
                }
            })
            .collect()
    }
}

impl<E> Env for RescaleObservation<E>
where
    E: Env<Obs = Vec<f32>, ObsSpace = BoundedSpace>,
{
    type Obs = Vec<f32>;
    type Act = E::Act;
    type ObsSpace = BoundedSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let result = self.env.step(action)?;
        Ok(StepResult {
            obs: self.rescale(&result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info,
        })
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
        let result = self.env.reset(seed)?;
        Ok(ResetResult {
            obs: self.rescale(&result.obs),
            info: result.info,
        })
    }

    fn observation_space(&self) -> &Self::ObsSpace {
        &self.new_space
    }

    delegate_env!(env, render, close, render_mode, action_space);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{PendulumConfig, PendulumEnv};
    use crate::space::Space;

    // Use Pendulum because its observation space has finite bounds,
    // unlike CartPole which has ±inf velocity bounds.

    #[test]
    fn rescales_observations() {
        let env = PendulumEnv::new(PendulumConfig::default()).unwrap();
        let mut env = RescaleObservation::new(env, 0.0, 1.0).unwrap();
        let r = env.reset(Some(42)).unwrap();
        assert_eq!(r.obs.len(), 3);
        for &v in &r.obs {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn observation_space_reflects_new_bounds() {
        let env = PendulumEnv::new(PendulumConfig::default()).unwrap();
        let env = RescaleObservation::new(env, -1.0, 1.0).unwrap();
        let space = env.observation_space();
        for &lo in &space.low {
            assert!((lo - (-1.0)).abs() < f32::EPSILON);
        }
        for &hi in &space.high {
            assert!((hi - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn rejects_invalid_range() {
        let env = PendulumEnv::new(PendulumConfig::default()).unwrap();
        assert!(RescaleObservation::new(env, 1.0, 0.0).is_err());
    }

    #[test]
    fn step_produces_rescaled_obs() {
        let env = PendulumEnv::new(PendulumConfig::default()).unwrap();
        let mut env = RescaleObservation::new(env, -10.0, 10.0).unwrap();
        env.reset(Some(42)).unwrap();
        let s = env.step(&vec![0.0]).unwrap();
        assert_eq!(s.obs.len(), 3);
        let space = env.observation_space();
        assert!(space.contains(&s.obs));
    }
}
