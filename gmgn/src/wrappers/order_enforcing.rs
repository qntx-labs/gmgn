//! Enforces the correct call order: `reset` must be called before `step` or `render`.
//!
//! Mirrors [Gymnasium `OrderEnforcing`](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.OrderEnforcing).

use crate::env::{Env, ResetResult, StepResult};
use crate::error::{Error, Result};
use crate::macros::delegate_env;

/// Ensures that [`reset`](Env::reset) is called before [`step`](Env::step) or
/// [`render`](Env::render).
///
/// Calling `step` or `render` before `reset` returns
/// [`Error::ResetNeeded`] instead of producing undefined behaviour or a
/// panic in the inner environment.
///
/// # Examples
///
/// ```rust
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::OrderEnforcing;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = OrderEnforcing::new(env);
/// // env.step(&0) would return Err(Error::ResetNeeded { .. })
/// let _reset = env.reset(Some(42)).unwrap();
/// let _step = env.step(&0).unwrap(); // OK after reset
/// ```
#[derive(Debug)]
pub struct OrderEnforcing<E: Env> {
    env: E,
    has_reset: bool,
}

impl<E: Env> OrderEnforcing<E> {
    /// Wrap `env` with call-order enforcement.
    #[must_use]
    pub const fn new(env: E) -> Self {
        Self {
            env,
            has_reset: false,
        }
    }

    /// Whether [`reset`](Env::reset) has been called at least once.
    #[must_use]
    pub const fn has_reset(&self) -> bool {
        self.has_reset
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

impl<E: Env> Env for OrderEnforcing<E> {
    type Obs = E::Obs;
    type Act = E::Act;
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        if !self.has_reset {
            return Err(Error::ResetNeeded { method: "step" });
        }
        self.env.step(action)
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
        self.has_reset = true;
        self.env.reset(seed)
    }

    fn render(&mut self) -> Result<crate::env::RenderFrame> {
        if !self.has_reset {
            return Err(Error::ResetNeeded { method: "render" });
        }
        self.env.render()
    }

    delegate_env!(env, close, render_mode, observation_space, action_space);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};

    #[test]
    fn step_before_reset_errors() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = OrderEnforcing::new(env);
        let result = env.step(&0);
        assert!(result.is_err());
    }

    #[test]
    fn step_after_reset_ok() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = OrderEnforcing::new(env);
        env.reset(Some(42)).unwrap();
        let result = env.step(&0);
        assert!(result.is_ok());
    }

    #[test]
    fn render_before_reset_errors() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = OrderEnforcing::new(env);
        let result = env.render();
        assert!(result.is_err());
    }

    #[test]
    fn has_reset_tracks_state() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = OrderEnforcing::new(env);
        assert!(!env.has_reset());
        env.reset(Some(0)).unwrap();
        assert!(env.has_reset());
    }
}
