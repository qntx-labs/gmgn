//! Applies an arbitrary function to actions before passing them to the inner env.
//!
//! Mirrors [Gymnasium `TransformAction`](https://gymnasium.farama.org/api/wrappers/action_wrappers/#gymnasium.wrappers.TransformAction).

use crate::env::{Env, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::Result;
use crate::space::Space;

/// Applies a user-supplied function `f(action) -> inner_action` to every
/// action before forwarding it to the inner environment.
///
/// The wrapper's action space type changes to `NewActSpace`.
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::TransformAction;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let act_space = env.action_space().clone();
/// // Identity transform — just demonstrating the API.
/// let mut env = TransformAction::new(env, |a: &i64| *a, act_space);
/// let _reset = env.reset(Some(42)).unwrap();
/// let _step = env.step(&0).unwrap();
/// ```
pub struct TransformAction<E, NewAct, NewActSpace, F>
where
    E: Env,
    NewActSpace: Space<Element = NewAct>,
    F: Fn(&NewAct) -> E::Act,
{
    env: E,
    f: F,
    act_space: NewActSpace,
}

impl<E, NewAct, NewActSpace, F> std::fmt::Debug for TransformAction<E, NewAct, NewActSpace, F>
where
    E: Env + std::fmt::Debug,
    NewActSpace: Space<Element = NewAct> + std::fmt::Debug,
    F: Fn(&NewAct) -> E::Act,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("TransformAction")
            .field("env", &self.env)
            .field("act_space", &self.act_space)
            .finish_non_exhaustive()
    }
}

impl<E, NewAct, NewActSpace, F> TransformAction<E, NewAct, NewActSpace, F>
where
    E: Env,
    NewActSpace: Space<Element = NewAct>,
    F: Fn(&NewAct) -> E::Act,
{
    /// Wrap `env` so that actions are transformed by `f` before being applied.
    ///
    /// `act_space` describes the new action space exposed by this wrapper.
    #[must_use]
    pub const fn new(env: E, f: F, act_space: NewActSpace) -> Self {
        Self { env, f, act_space }
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

impl<E, NewAct, NewActSpace, F> Env for TransformAction<E, NewAct, NewActSpace, F>
where
    E: Env,
    NewActSpace: Space<Element = NewAct>,
    F: Fn(&NewAct) -> E::Act,
{
    type Obs = E::Obs;
    type Act = NewAct;
    type ObsSpace = E::ObsSpace;
    type ActSpace = NewActSpace;

    fn step(&mut self, action: &NewAct) -> Result<StepResult<Self::Obs>> {
        let inner_action = (self.f)(action);
        self.env.step(&inner_action)
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
        self.env.reset(seed)
    }

    fn render(&mut self) -> Result<RenderFrame> {
        self.env.render()
    }

    fn close(&mut self) {
        self.env.close();
    }

    fn observation_space(&self) -> &Self::ObsSpace {
        self.env.observation_space()
    }

    fn action_space(&self) -> &Self::ActSpace {
        &self.act_space
    }

    fn render_mode(&self) -> &RenderMode {
        self.env.render_mode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};

    #[test]
    fn identity_transform() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let act_space = *env.action_space();
        let mut env = TransformAction::new(env, |a: &i64| *a, act_space);
        env.reset(Some(42)).unwrap();
        let r = env.step(&0).unwrap();
        assert_eq!(r.obs.len(), 4);
    }

    #[test]
    fn flip_action() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let act_space = *env.action_space();
        // Flip: 0 -> 1, 1 -> 0.
        let mut env = TransformAction::new(env, |a: &i64| 1 - a, act_space);
        env.reset(Some(42)).unwrap();
        let r = env.step(&0).unwrap(); // inner env receives action 1
        assert_eq!(r.obs.len(), 4);
    }
}
