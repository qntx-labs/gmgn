//! Applies an arbitrary function to observations.
//!
//! Mirrors [Gymnasium `TransformObservation`](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.TransformObservation).

use crate::env::{Env, ResetResult, StepResult};
use crate::error::Result;
use crate::macros::delegate_env;
use crate::space::Space;

/// Applies a user-supplied function `f(obs) -> new_obs` to every observation
/// returned by [`step`](Env::step) and [`reset`](Env::reset).
///
/// The observation space type changes to `NewObsSpace` which must be provided
/// together with the transform function.
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::TransformObservation;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let obs_space = env.observation_space().clone();
/// let mut env = TransformObservation::new(
///     env,
///     |obs: Vec<f32>| obs.iter().map(|x| x * 2.0).collect::<Vec<f32>>(),
///     obs_space,
/// );
/// let reset = env.reset(Some(42)).unwrap();
/// ```
pub struct TransformObservation<E, NewObs, NewObsSpace, F>
where
    E: Env,
    NewObsSpace: Space<Element = NewObs>,
    F: Fn(E::Obs) -> NewObs,
{
    env: E,
    f: F,
    obs_space: NewObsSpace,
}

impl<E, NewObs, NewObsSpace, F> std::fmt::Debug for TransformObservation<E, NewObs, NewObsSpace, F>
where
    E: Env + std::fmt::Debug,
    NewObsSpace: Space<Element = NewObs> + std::fmt::Debug,
    F: Fn(E::Obs) -> NewObs,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("TransformObservation")
            .field("env", &self.env)
            .field("obs_space", &self.obs_space)
            .finish_non_exhaustive()
    }
}

impl<E, NewObs, NewObsSpace, F> TransformObservation<E, NewObs, NewObsSpace, F>
where
    E: Env,
    NewObsSpace: Space<Element = NewObs>,
    F: Fn(E::Obs) -> NewObs,
{
    /// Wrap `env` so that observations are transformed by `f`.
    ///
    /// `obs_space` describes the output space after transformation.
    #[must_use]
    pub const fn new(env: E, f: F, obs_space: NewObsSpace) -> Self {
        Self { env, f, obs_space }
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

impl<E, NewObs, NewObsSpace, F> Env for TransformObservation<E, NewObs, NewObsSpace, F>
where
    E: Env,
    NewObsSpace: Space<Element = NewObs>,
    F: Fn(E::Obs) -> NewObs,
{
    type Obs = NewObs;
    type Act = E::Act;
    type ObsSpace = NewObsSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let result = self.env.step(action)?;
        Ok(StepResult {
            obs: (self.f)(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info,
        })
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
        let result = self.env.reset(seed)?;
        Ok(ResetResult {
            obs: (self.f)(result.obs),
            info: result.info,
        })
    }

    fn observation_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    delegate_env!(env, render, close, render_mode, action_space);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};

    #[test]
    fn transforms_step_observation() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let obs_space = env.observation_space().clone();
        let mut env = TransformObservation::new(
            env,
            |obs: Vec<f32>| obs.iter().map(|x| x * 2.0).collect::<Vec<f32>>(),
            obs_space,
        );
        env.reset(Some(42)).unwrap();
        let original_env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        // Just verify it runs and returns 4 elements.
        let r = env.step(&0).unwrap();
        assert_eq!(r.obs.len(), 4);
        let _ = original_env;
    }

    #[test]
    fn transforms_reset_observation() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let obs_space = env.observation_space().clone();
        let mut env =
            TransformObservation::new(env, |obs: Vec<f32>| vec![obs.len() as f32], obs_space);
        let r = env.reset(Some(42)).unwrap();
        // Transform maps obs to [4.0] (length of original obs).
        assert_eq!(r.obs, vec![4.0]);
    }
}
