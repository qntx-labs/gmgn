//! Clips continuous actions to the action space bounds.
//!
//! Mirrors [Gymnasium `ClipAction`](https://gymnasium.farama.org/api/wrappers/action_wrappers/#gymnasium.wrappers.ClipAction).

use crate::env::{Env, StepResult};
use crate::error::Result;
use crate::macros::delegate_env;
use crate::space::BoundedSpace;

/// Clips every element of a continuous action to the bounds defined by the
/// environment's [`BoundedSpace`] action space before forwarding to `step`.
///
/// Only applicable to environments whose `ActSpace` is [`BoundedSpace`] and
/// `Act` is `Vec<f32>`.
///
/// # Examples
///
/// ```rust,ignore
/// use gmgn::wrappers::ClipAction;
///
/// let env = SomeContinuousEnv::new(config)?;
/// let mut env = ClipAction::new(env);
/// ```
#[derive(Debug)]
pub struct ClipAction<E>
where
    E: Env<Act = Vec<f32>, ActSpace = BoundedSpace>,
{
    env: E,
}

impl<E> ClipAction<E>
where
    E: Env<Act = Vec<f32>, ActSpace = BoundedSpace>,
{
    /// Wrap `env` so that all actions are clipped to the action space bounds.
    #[must_use]
    pub const fn new(env: E) -> Self {
        Self { env }
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

    /// Clip `action` element-wise to `[low, high]`.
    fn clip(&self, action: &[f32]) -> Vec<f32> {
        let space = self.env.action_space();
        action
            .iter()
            .zip(space.low.iter().zip(space.high.iter()))
            .map(|(&a, (&lo, &hi))| a.clamp(lo, hi))
            .collect()
    }
}

impl<E> Env for ClipAction<E>
where
    E: Env<Act = Vec<f32>, ActSpace = BoundedSpace>,
{
    type Obs = E::Obs;
    type Act = Vec<f32>;
    type ObsSpace = E::ObsSpace;
    type ActSpace = BoundedSpace;

    fn step(&mut self, action: &Vec<f32>) -> Result<StepResult<Self::Obs>> {
        let clipped = self.clip(action);
        self.env.step(&clipped)
    }

    delegate_env!(
        env,
        reset,
        render,
        close,
        render_mode,
        observation_space,
        action_space
    );
}
