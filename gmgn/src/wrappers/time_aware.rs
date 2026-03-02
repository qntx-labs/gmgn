//! Appends the current timestep to the observation.
//!
//! Mirrors [Gymnasium `TimeAwareObservation`](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.TimeAwareObservation).

use crate::env::{Env, ResetResult, StepResult};
use crate::error::Result;
use crate::macros::delegate_env;
use crate::space::BoundedSpace;

/// Appends the current timestep (normalized or raw) to the observation vector.
///
/// For an environment with observation dimension `D`, the wrapped observation
/// has dimension `D + 1`. The appended value is `step_count / max_steps` when
/// `max_steps` is provided, or the raw step count otherwise.
///
/// Only applicable to environments whose `Obs` is `Vec<f32>` and `ObsSpace`
/// is [`BoundedSpace`].
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::TimeAwareObservation;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = TimeAwareObservation::new(env, Some(500));
/// let r = env.reset(Some(42)).unwrap();
/// // CartPole obs dim = 4, plus 1 time feature → 5
/// assert_eq!(r.obs.len(), 5);
/// ```
#[derive(Debug)]
pub struct TimeAwareObservation<E>
where
    E: Env<Obs = Vec<f32>, ObsSpace = BoundedSpace>,
{
    env: E,
    step_count: u64,
    max_steps: Option<u64>,
    obs_space: BoundedSpace,
}

impl<E> TimeAwareObservation<E>
where
    E: Env<Obs = Vec<f32>, ObsSpace = BoundedSpace>,
{
    /// Wrap `env` to append a time feature to each observation.
    ///
    /// If `max_steps` is `Some(n)`, the time feature is normalized to `[0, 1]`
    /// as `step_count / n`. Otherwise the raw step count (cast to f32) is used.
    #[must_use]
    pub fn new(env: E, max_steps: Option<u64>) -> Self {
        let inner_space = env.observation_space();
        let mut low = inner_space.low.clone();
        let mut high = inner_space.high.clone();
        low.push(0.0);
        high.push(if max_steps.is_some() {
            1.0
        } else {
            f32::INFINITY
        });
        // Safety: extending valid bounds with [0, ∞) or [0, 1] always produces valid bounds.
        let obs_space = BoundedSpace::new(low, high)
            .unwrap_or_else(|_| unreachable!("extending valid bounds cannot fail"));

        Self {
            env,
            step_count: 0,
            max_steps,
            obs_space,
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

    /// Compute the time feature value.
    #[allow(clippy::cast_possible_truncation)]
    fn time_feature(&self) -> f32 {
        self.max_steps.map_or(self.step_count as f32, |max| {
            (self.step_count as f64 / max as f64) as f32
        })
    }

    /// Append the time feature to an observation.
    fn augment(&self, mut obs: Vec<f32>) -> Vec<f32> {
        obs.push(self.time_feature());
        obs
    }
}

impl<E> Env for TimeAwareObservation<E>
where
    E: Env<Obs = Vec<f32>, ObsSpace = BoundedSpace>,
{
    type Obs = Vec<f32>;
    type Act = E::Act;
    type ObsSpace = BoundedSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let result = self.env.step(action)?;
        self.step_count += 1;
        Ok(StepResult {
            obs: self.augment(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info,
        })
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
        self.step_count = 0;
        let result = self.env.reset(seed)?;
        Ok(ResetResult {
            obs: self.augment(result.obs),
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
    fn appends_time_feature() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = TimeAwareObservation::new(env, Some(500));
        let r = env.reset(Some(42)).unwrap();
        assert_eq!(r.obs.len(), 5);
        // After reset, time = 0.0.
        assert!((r.obs[4] - 0.0).abs() < f32::EPSILON);

        let s = env.step(&0).unwrap();
        assert_eq!(s.obs.len(), 5);
        // After 1 step, time = 1/500 = 0.002.
        assert!((s.obs[4] - 0.002).abs() < 1e-6);
    }

    #[test]
    fn raw_step_count_without_max() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = TimeAwareObservation::new(env, None);
        env.reset(Some(42)).unwrap();
        let s = env.step(&0).unwrap();
        assert!((s.obs[4] - 1.0).abs() < f32::EPSILON);
    }
}
