//! Repeats the previous action with a given probability.
//!
//! Mirrors [Gymnasium `StickyAction`](https://gymnasium.farama.org/api/wrappers/action_wrappers/#gymnasium.wrappers.StickyAction).

use crate::env::{Env, ResetResult, StepResult};
use crate::error::Result;
use crate::macros::delegate_env;

/// With probability `repeat_action_probability`, the previous action is
/// repeated instead of applying the new one.
///
/// This wrapper follows [Machado et al., 2018](https://arxiv.org/pdf/1709.06009.pdf)
/// and is commonly used in Atari environments to introduce stochasticity.
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::StickyAction;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = StickyAction::new(env, 0.25).unwrap();
/// ```
#[derive(Debug)]
pub struct StickyAction<E: Env>
where
    E::Act: Clone,
{
    env: E,
    repeat_probability: f64,
    last_action: Option<E::Act>,
    rng: crate::rng::Rng,
}

impl<E: Env> StickyAction<E>
where
    E::Act: Clone,
{
    /// Wrap `env` with sticky action behaviour.
    ///
    /// `repeat_probability` must be in `[0, 1)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `repeat_probability` is not in `[0, 1)`.
    pub fn new(env: E, repeat_probability: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&repeat_probability) {
            return Err(crate::error::Error::InvalidAction {
                reason: format!(
                    "repeat_probability must be in [0, 1), got {repeat_probability}"
                ),
            });
        }
        Ok(Self {
            env,
            repeat_probability,
            last_action: None,
            rng: crate::rng::create_rng(None),
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

impl<E: Env> Env for StickyAction<E>
where
    E::Act: Clone,
{
    type Obs = E::Obs;
    type Act = E::Act;
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        use rand::RngExt;
        let effective_action = if let Some(ref prev) = self.last_action {
            let r: f64 = self.rng.random();
            if r < self.repeat_probability {
                prev.clone()
            } else {
                action.clone()
            }
        } else {
            action.clone()
        };
        self.last_action = Some(effective_action.clone());
        self.env.step(&effective_action)
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
        self.last_action = None;
        if let Some(s) = seed {
            self.rng = crate::rng::create_rng(Some(s));
        }
        self.env.reset(seed)
    }

    delegate_env!(env, render, close, render_mode, observation_space, action_space);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};

    #[test]
    fn zero_probability_passes_through() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = StickyAction::new(env, 0.0).unwrap();
        env.reset(Some(42)).unwrap();
        let r = env.step(&1).unwrap();
        assert_eq!(r.obs.len(), 4);
    }

    #[test]
    fn high_probability_repeats() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = StickyAction::new(env, 0.99).unwrap();
        env.reset(Some(42)).unwrap();
        // First step sets last_action.
        env.step(&0).unwrap();
        // Subsequent steps almost certainly repeat action 0.
        for _ in 0..10 {
            let r = env.step(&1).unwrap();
            assert_eq!(r.obs.len(), 4);
            if r.terminated {
                env.reset(None).unwrap();
            }
        }
    }

    #[test]
    fn rejects_invalid_probability() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        assert!(StickyAction::new(env, 1.0).is_err());
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        assert!(StickyAction::new(env, -0.1).is_err());
    }
}
