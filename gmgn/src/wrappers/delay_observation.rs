//! Delays returned observations by a fixed number of timesteps.
//!
//! Mirrors [Gymnasium `DelayObservation`](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.DelayObservation).

use std::collections::VecDeque;

use crate::env::{Env, ResetResult, StepResult};
use crate::error::Result;
use crate::macros::delegate_env;

/// Delays the observation returned by the environment by `delay` timesteps.
///
/// Before enough timesteps have elapsed, the wrapper returns a zero-like
/// clone of the first observation seen (with all `f32` values set to 0).
///
/// # Type Constraints
///
/// `Obs` must implement [`Clone`] (observations are buffered) and the
/// caller must supply a `zero_fn` that produces a blank observation.
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::DelayObservation;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = DelayObservation::new(env, 2, || vec![0.0; 4]);
/// let reset = env.reset(Some(42)).unwrap();
/// // First observation is zero-filled because of the delay.
/// assert_eq!(reset.obs, vec![0.0; 4]);
/// ```
#[derive(Debug)]
pub struct DelayObservation<E: Env>
where
    E::Obs: Clone,
{
    env: E,
    /// Number of timesteps to delay.
    delay: usize,
    /// Queue holding buffered observations.
    queue: VecDeque<E::Obs>,
    /// Factory that produces a zero / placeholder observation.
    zero_fn: fn() -> E::Obs,
}

impl<E: Env> DelayObservation<E>
where
    E::Obs: Clone,
{
    /// Wrap `env` with an observation delay of `delay` timesteps.
    ///
    /// `zero_fn` is called to produce the placeholder observation returned
    /// before the delay window is filled (e.g. `|| vec![0.0; 4]`).
    ///
    /// A delay of 0 is a no-op (observations pass through unchanged).
    #[must_use]
    pub fn new(env: E, delay: usize, zero_fn: fn() -> E::Obs) -> Self {
        Self {
            env,
            delay,
            queue: VecDeque::with_capacity(delay + 1),
            zero_fn,
        }
    }

    /// The delay in timesteps.
    #[must_use]
    pub const fn delay(&self) -> usize {
        self.delay
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

    /// Process an observation through the delay queue.
    fn delayed_obs(&mut self, obs: E::Obs) -> E::Obs {
        self.queue.push_back(obs);
        if self.queue.len() > self.delay {
            self.queue.pop_front().expect("queue non-empty")
        } else {
            (self.zero_fn)()
        }
    }
}

impl<E: Env> Env for DelayObservation<E>
where
    E::Obs: Clone,
{
    type Obs = E::Obs;
    type Act = E::Act;
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let mut r = self.env.step(action)?;
        r.obs = self.delayed_obs(r.obs);
        Ok(r)
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
        self.queue.clear();
        let mut r = self.env.reset(seed)?;
        r.obs = self.delayed_obs(r.obs);
        Ok(r)
    }

    delegate_env!(env);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};
    use crate::space::Space;

    #[test]
    fn delay_zero_is_passthrough() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut direct = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut delayed = DelayObservation::new(env, 0, || vec![0.0; 4]);

        let r1 = direct.reset(Some(42)).unwrap();
        let r2 = delayed.reset(Some(42)).unwrap();
        assert_eq!(r1.obs, r2.obs);
    }

    #[test]
    fn delay_returns_zeros_initially() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut delayed = DelayObservation::new(env, 2, || vec![0.0_f32; 4]);
        let r = delayed.reset(Some(42)).unwrap();
        assert_eq!(r.obs, vec![0.0_f32; 4]);

        let s1 = delayed.step(&0).unwrap();
        assert_eq!(s1.obs, vec![0.0_f32; 4]);
    }

    #[test]
    fn delay_returns_real_obs_after_delay() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut direct = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut delayed = DelayObservation::new(env, 1, || vec![0.0_f32; 4]);

        let direct_reset = direct.reset(Some(42)).unwrap();
        let _ = delayed.reset(Some(42)).unwrap();

        // After 1 step, the delay=1 wrapper should return the reset observation.
        let s = delayed.step(&0).unwrap();
        assert_eq!(s.obs, direct_reset.obs);
    }

    #[test]
    fn reset_clears_queue() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut delayed = DelayObservation::new(env, 1, || vec![0.0_f32; 4]);
        delayed.reset(Some(42)).unwrap();
        delayed.step(&0).unwrap();

        // Reset should clear the queue — next obs should be zeros again.
        let r = delayed.reset(Some(99)).unwrap();
        assert_eq!(r.obs, vec![0.0_f32; 4]);
    }

    #[test]
    fn delegates_spaces() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let wrapped = DelayObservation::new(env, 3, || vec![0.0_f32; 4]);
        assert_eq!(wrapped.observation_space().shape(), &[4]);
        assert_eq!(wrapped.action_space().n, 2);
    }
}
