//! Rescales continuous actions from one bounded range to another.
//!
//! Mirrors [Gymnasium `RescaleAction`](https://gymnasium.farama.org/api/wrappers/action_wrappers/#gymnasium.wrappers.RescaleAction).

use crate::env::{Env, StepResult};
use crate::error::{Error, Result};
use crate::macros::delegate_env;
use crate::space::BoundedSpace;

/// Rescales continuous actions from `[min_action, max_action]` to the inner
/// environment's action space bounds.
///
/// Only works with environments whose action space is [`BoundedSpace`].
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{PendulumEnv, PendulumConfig};
/// use gmgn::wrappers::RescaleAction;
///
/// let env = PendulumEnv::new(PendulumConfig::default()).unwrap();
/// let mut env = RescaleAction::new(env, -1.0, 1.0).unwrap();
/// let _reset = env.reset(Some(42)).unwrap();
/// let _step = env.step(&vec![0.5]).unwrap(); // mapped to inner bounds
/// ```
#[derive(Debug)]
pub struct RescaleAction<E: Env<Act = Vec<f32>, ActSpace = BoundedSpace>> {
    env: E,
    new_space: BoundedSpace,
    min_action: Vec<f32>,
    max_action: Vec<f32>,
}

impl<E: Env<Act = Vec<f32>, ActSpace = BoundedSpace>> RescaleAction<E> {
    /// Wrap `env` so that actions in `[min_action, max_action]` are linearly
    /// mapped to the inner environment's action bounds.
    ///
    /// # Errors
    ///
    /// Returns an error if `min_action >= max_action` or if the inner
    /// action space has non-finite bounds.
    pub fn new(env: E, min_action: f32, max_action: f32) -> Result<Self> {
        if min_action >= max_action {
            return Err(Error::InvalidAction {
                reason: format!("min_action ({min_action}) >= max_action ({max_action})"),
            });
        }

        let inner_space = env.action_space();
        let dim = inner_space.low.len();

        // Verify inner bounds are finite.
        for (i, (lo, hi)) in inner_space
            .low
            .iter()
            .zip(inner_space.high.iter())
            .enumerate()
        {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(Error::InvalidAction {
                    reason: format!(
                        "inner action space dim {i} has non-finite bounds [{lo}, {hi}]"
                    ),
                });
            }
        }

        let new_space = BoundedSpace::uniform(min_action, max_action, dim)?;
        let min_vec = vec![min_action; dim];
        let max_vec = vec![max_action; dim];

        Ok(Self {
            env,
            new_space,
            min_action: min_vec,
            max_action: max_vec,
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
}

impl<E: Env<Act = Vec<f32>, ActSpace = BoundedSpace>> Env for RescaleAction<E> {
    type Obs = E::Obs;
    type Act = Vec<f32>;
    type ObsSpace = E::ObsSpace;
    type ActSpace = BoundedSpace;

    fn step(&mut self, action: &Vec<f32>) -> Result<StepResult<Self::Obs>> {
        // Linearly map from [min_action, max_action] to [inner_low, inner_high].
        let inner_space = self.env.action_space();
        let mapped: Vec<f32> = action
            .iter()
            .zip(self.min_action.iter().zip(self.max_action.iter()))
            .zip(inner_space.low.iter().zip(inner_space.high.iter()))
            .map(|((&a, (&new_lo, &new_hi)), (&old_lo, &old_hi))| {
                let t = (a - new_lo) / (new_hi - new_lo);
                t.mul_add(old_hi - old_lo, old_lo)
            })
            .collect();
        self.env.step(&mapped)
    }

    fn action_space(&self) -> &Self::ActSpace {
        &self.new_space
    }

    delegate_env!(env, reset, render, close, render_mode, observation_space);
}

#[cfg(test)]
#[allow(clippy::panic)] // Panics are acceptable in test assertions.
mod tests {
    use super::*;
    use crate::envs::classic_control::{PendulumConfig, PendulumEnv};
    use crate::space::{Space, SpaceInfo};

    #[test]
    fn rescale_maps_actions() {
        // Pendulum action space is [-2.0, 2.0].
        let env = PendulumEnv::new(PendulumConfig::default()).unwrap();
        let mut env = RescaleAction::new(env, -1.0, 1.0).unwrap();

        assert_eq!(env.action_space().low, vec![-1.0]);
        assert_eq!(env.action_space().high, vec![1.0]);

        env.reset(Some(42)).unwrap();
        // Action 0.0 in [-1, 1] should map to 0.0 in [-2, 2].
        let r = env.step(&vec![0.0]);
        assert!(r.is_ok());
    }

    #[test]
    fn rejects_invalid_range() {
        let env = PendulumEnv::new(PendulumConfig::default()).unwrap();
        let result = RescaleAction::new(env, 1.0, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn space_info_reflects_new_bounds() {
        let env = PendulumEnv::new(PendulumConfig::default()).unwrap();
        let env = RescaleAction::new(env, 0.0, 1.0).unwrap();
        match env.action_space().space_info() {
            SpaceInfo::Bounded { low, high, .. } => {
                assert_eq!(low, vec![0.0]);
                assert_eq!(high, vec![1.0]);
            }
            other => panic!("expected Bounded, got {other:?}"),
        }
    }
}
