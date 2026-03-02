//! Environment wrapper for modular behavior transformation.
//!
//! [`Wrapper`] delegates all [`Env`] methods to an inner environment,
//! allowing subsets of behavior to be overridden by composing wrappers.

use crate::env::{Env, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::Result;

/// A transparent wrapper around an [`Env`] implementation.
///
/// By default, every method delegates to the inner environment. Override
/// individual methods to customize behavior (e.g. reward shaping, observation
/// transforms, time limits).
///
/// # Examples
///
/// ```rust,ignore
/// struct TimeLimit<E: Env> {
///     inner: Wrapper<E>,
///     max_steps: u64,
///     elapsed: u64,
/// }
/// ```
#[derive(Debug)]
pub struct Wrapper<E: Env> {
    /// The wrapped environment.
    pub env: E,
}

impl<E: Env> Wrapper<E> {
    /// Wrap an existing environment.
    #[must_use]
    pub const fn new(env: E) -> Self {
        Self { env }
    }

    /// Unwrap and return the inner environment.
    #[must_use]
    pub fn into_inner(self) -> E {
        self.env
    }

    /// Get a reference to the innermost (unwrapped) environment.
    #[must_use]
    pub const fn unwrapped(&self) -> &E {
        &self.env
    }
}

impl<E: Env> Env for Wrapper<E> {
    type Obs = E::Obs;
    type Act = E::Act;
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        self.env.step(action)
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
        self.env.action_space()
    }

    fn render_mode(&self) -> &RenderMode {
        self.env.render_mode()
    }
}
