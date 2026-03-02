//! Stacks consecutive observations into a single observation.
//!
//! Mirrors [Gymnasium `FrameStackObservation`](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.FrameStackObservation).

use std::collections::VecDeque;

use crate::env::{Env, ResetResult, StepResult};
use crate::error::Result;
use crate::macros::delegate_env;
use crate::space::BoundedSpace;

/// Stacks the last `num_stack` observations into a single flat `Vec<f32>`.
///
/// For an environment with observation dimension `D`, the wrapped observation
/// has dimension `D * num_stack`. On reset the initial observation is repeated
/// `num_stack` times to fill the buffer.
///
/// Only applicable to environments whose `Obs` is `Vec<f32>` and `ObsSpace`
/// is [`BoundedSpace`].
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::FrameStackObservation;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = FrameStackObservation::new(env, 4).unwrap();
/// let reset = env.reset(Some(42)).unwrap();
/// // CartPole obs dim = 4, stacked 4 times → 16
/// assert_eq!(reset.obs.len(), 16);
/// ```
#[derive(Debug)]
pub struct FrameStackObservation<E>
where
    E: Env<Obs = Vec<f32>, ObsSpace = BoundedSpace>,
{
    env: E,
    /// Number of frames to stack.
    num_stack: usize,
    /// Ring buffer of recent observations (front = oldest).
    frames: VecDeque<Vec<f32>>,
    /// Stacked observation space with expanded bounds.
    stacked_space: BoundedSpace,
}

impl<E> FrameStackObservation<E>
where
    E: Env<Obs = Vec<f32>, ObsSpace = BoundedSpace>,
{
    /// Wrap `env` to stack the last `num_stack` observations.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_stack` is zero.
    pub fn new(env: E, num_stack: usize) -> Result<Self> {
        if num_stack == 0 {
            return Err(crate::error::Error::InvalidSpace {
                reason: "num_stack must be >= 1".to_owned(),
            });
        }

        let inner_space = env.observation_space();
        let dim = inner_space.low.len();

        // Repeat low/high bounds num_stack times.
        let low: Vec<f32> = inner_space.low.iter().cycle().take(dim * num_stack).copied().collect();
        let high: Vec<f32> = inner_space.high.iter().cycle().take(dim * num_stack).copied().collect();
        let stacked_space = BoundedSpace::new(low, high).map_err(|e| {
            crate::error::Error::InvalidSpace {
                reason: format!("failed to create stacked observation space: {e}"),
            }
        })?;

        Ok(Self {
            env,
            num_stack,
            frames: VecDeque::with_capacity(num_stack),
            stacked_space,
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

    /// Concatenate all frames in the buffer into a single flat vector.
    fn stacked_obs(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.stacked_space.low.len());
        for frame in &self.frames {
            out.extend_from_slice(frame);
        }
        out
    }
}

impl<E> Env for FrameStackObservation<E>
where
    E: Env<Obs = Vec<f32>, ObsSpace = BoundedSpace>,
{
    type Obs = Vec<f32>;
    type Act = E::Act;
    type ObsSpace = BoundedSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let result = self.env.step(action)?;
        self.frames.pop_front();
        self.frames.push_back(result.obs);
        Ok(StepResult {
            obs: self.stacked_obs(),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info,
        })
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
        let result = self.env.reset(seed)?;
        self.frames.clear();
        for _ in 0..self.num_stack {
            self.frames.push_back(result.obs.clone());
        }
        Ok(ResetResult {
            obs: self.stacked_obs(),
            info: result.info,
        })
    }

    fn observation_space(&self) -> &Self::ObsSpace {
        &self.stacked_space
    }

    delegate_env!(env, render, close, render_mode, action_space);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};

    #[test]
    fn stacks_observations() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let obs_dim = env.observation_space().low.len(); // 4
        let num_stack = 3;
        let mut env = FrameStackObservation::new(env, num_stack).unwrap();

        let r = env.reset(Some(42)).unwrap();
        assert_eq!(r.obs.len(), obs_dim * num_stack);

        // After reset, all frames are identical.
        let frame = &r.obs[..obs_dim];
        for i in 1..num_stack {
            assert_eq!(&r.obs[i * obs_dim..(i + 1) * obs_dim], frame);
        }
    }

    #[test]
    fn step_shifts_frames() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let obs_dim = env.observation_space().low.len();
        let num_stack = 2;
        let mut env = FrameStackObservation::new(env, num_stack).unwrap();

        let r = env.reset(Some(42)).unwrap();
        let initial_frame: Vec<f32> = r.obs[..obs_dim].to_vec();

        let s = env.step(&0).unwrap();
        assert_eq!(s.obs.len(), obs_dim * num_stack);
        // First half should be the initial frame, second half the new frame.
        assert_eq!(&s.obs[..obs_dim], &initial_frame[..]);
        // New frame should differ (cart moved).
        assert_ne!(&s.obs[obs_dim..], &initial_frame[..]);
    }

    #[test]
    fn observation_space_matches() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let obs_dim = env.observation_space().low.len();
        let num_stack = 4;
        let env = FrameStackObservation::new(env, num_stack).unwrap();

        assert_eq!(env.observation_space().low.len(), obs_dim * num_stack);
        assert_eq!(env.observation_space().high.len(), obs_dim * num_stack);
    }

    #[test]
    fn rejects_zero_stack() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        assert!(FrameStackObservation::new(env, 0).is_err());
    }
}
