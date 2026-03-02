//! Flattens observations into a flat `Vec<f32>` vector.
//!
//! Mirrors [Gymnasium `FlattenObservation`](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.FlattenObservation).
//!
//! This wrapper is essential for feeding structured observations (e.g. from
//! [`Discrete`](crate::space::Discrete) or [`MultiDiscrete`](crate::space::MultiDiscrete))
//! into neural network policies that expect a flat continuous input.

use crate::env::{Env, ResetResult, StepResult};
use crate::error::Result;
use crate::macros::delegate_env;
use crate::space::BoundedSpace;

/// A closure that flattens an observation of type `O` into a `Vec<f32>`.
///
/// Users provide this at construction time to define how their specific
/// observation type maps to a flat vector.
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::FlattenObservation;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let obs_space = BoundedSpace::uniform(f32::MIN, f32::MAX, 4).unwrap();
/// let mut env = FlattenObservation::new(env, |obs: Vec<f32>| obs, obs_space);
/// let reset = env.reset(Some(42)).unwrap();
/// assert_eq!(reset.obs.len(), 4);
/// ```
pub struct FlattenObservation<E, F>
where
    E: Env,
    F: Fn(E::Obs) -> Vec<f32>,
{
    env: E,
    flatten_fn: F,
    flat_obs_space: BoundedSpace,
}

impl<E, F> std::fmt::Debug for FlattenObservation<E, F>
where
    E: Env + std::fmt::Debug,
    F: Fn(E::Obs) -> Vec<f32>,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("FlattenObservation")
            .field("env", &self.env)
            .field("flat_obs_space", &self.flat_obs_space)
            .finish_non_exhaustive()
    }
}

impl<E, F> FlattenObservation<E, F>
where
    E: Env,
    F: Fn(E::Obs) -> Vec<f32>,
{
    /// Wrap `env` so that observations are flattened by `flatten_fn`.
    ///
    /// `flat_obs_space` is the [`BoundedSpace`] describing the flattened output.
    #[must_use]
    pub const fn new(env: E, flatten_fn: F, flat_obs_space: BoundedSpace) -> Self {
        Self {
            env,
            flatten_fn,
            flat_obs_space,
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

impl<E, F> Env for FlattenObservation<E, F>
where
    E: Env,
    F: Fn(E::Obs) -> Vec<f32>,
{
    type Obs = Vec<f32>;
    type Act = E::Act;
    type ObsSpace = BoundedSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let r = self.env.step(action)?;
        Ok(StepResult {
            obs: (self.flatten_fn)(r.obs),
            reward: r.reward,
            terminated: r.terminated,
            truncated: r.truncated,
            info: r.info,
        })
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
        let r = self.env.reset(seed)?;
        Ok(ResetResult {
            obs: (self.flatten_fn)(r.obs),
            info: r.info,
        })
    }

    fn observation_space(&self) -> &Self::ObsSpace {
        &self.flat_obs_space
    }

    delegate_env!(env, render, close, render_mode, action_space);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};
    use crate::space::Space;

    #[test]
    fn flattens_observation() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let flat_space = BoundedSpace::uniform(f32::MIN, f32::MAX, 4).unwrap();
        let mut env = FlattenObservation::new(env, |obs: Vec<f32>| obs, flat_space);

        let r = env.reset(Some(42)).unwrap();
        assert_eq!(r.obs.len(), 4);

        let r = env.step(&0).unwrap();
        assert_eq!(r.obs.len(), 4);
    }

    #[test]
    fn observation_space_is_flat() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let flat_space = BoundedSpace::uniform(-10.0, 10.0, 4).unwrap();
        let env = FlattenObservation::new(env, |obs: Vec<f32>| obs, flat_space);
        assert_eq!(env.observation_space().shape(), &[4]);
    }

    #[test]
    fn delegates_action_space() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let flat_space = BoundedSpace::uniform(f32::MIN, f32::MAX, 4).unwrap();
        let env = FlattenObservation::new(env, |obs: Vec<f32>| obs, flat_space);
        assert_eq!(env.action_space().n, 2);
    }
}
