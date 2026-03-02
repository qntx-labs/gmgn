//! Type-erased wrappers for `Box<dyn DynEnv>`.
//!
//! These mirror [`TimeLimit`](crate::wrappers::TimeLimit) and
//! [`OrderEnforcing`](crate::wrappers::OrderEnforcing) but operate on
//! trait objects so that [`make_with`](super::make_with) can automatically
//! wrap environments produced by the registry.

use super::DynEnv;
use super::DynValue;
use crate::env::{RenderFrame, ResetResult, StepResult};
use crate::error::{Error, Result};
use crate::space::SpaceInfo;

/// Type-erased equivalent of [`TimeLimit`](crate::wrappers::TimeLimit).
///
/// Wraps a `Box<dyn DynEnv>` and truncates episodes after a fixed number of
/// steps — exactly the same semantics as the generic wrapper.
pub struct DynTimeLimit {
    env: Box<dyn DynEnv>,
    max_episode_steps: u64,
    elapsed_steps: Option<u64>,
}

impl std::fmt::Debug for DynTimeLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynTimeLimit")
            .field("max_episode_steps", &self.max_episode_steps)
            .field("elapsed_steps", &self.elapsed_steps)
            .field("env", &self.env)
            .finish()
    }
}

impl DynTimeLimit {
    pub(crate) fn new(env: Box<dyn DynEnv>, max_episode_steps: u64) -> Self {
        Self {
            env,
            max_episode_steps,
            elapsed_steps: None,
        }
    }
}

impl DynEnv for DynTimeLimit {
    fn step_dyn(&mut self, action: &DynValue) -> Result<StepResult<DynValue>> {
        let mut result = self.env.step_dyn(action)?;

        let elapsed = self
            .elapsed_steps
            .as_mut()
            .expect("environment must be reset before step");
        *elapsed += 1;

        if *elapsed >= self.max_episode_steps {
            result.truncated = true;
        }

        Ok(result)
    }

    fn reset_dyn(&mut self, seed: Option<u64>) -> Result<ResetResult<DynValue>> {
        self.elapsed_steps = Some(0);
        self.env.reset_dyn(seed)
    }

    fn render_dyn(&mut self) -> Result<RenderFrame> {
        self.env.render_dyn()
    }

    fn close_dyn(&mut self) {
        self.env.close_dyn();
    }

    fn observation_space_info(&self) -> SpaceInfo {
        self.env.observation_space_info()
    }

    fn action_space_info(&self) -> SpaceInfo {
        self.env.action_space_info()
    }
}

/// Type-erased equivalent of [`OrderEnforcing`](crate::wrappers::OrderEnforcing).
///
/// Ensures [`reset_dyn`](DynEnv::reset_dyn) is called before
/// [`step_dyn`](DynEnv::step_dyn) or [`render_dyn`](DynEnv::render_dyn).
pub struct DynOrderEnforcing {
    env: Box<dyn DynEnv>,
    has_reset: bool,
}

impl std::fmt::Debug for DynOrderEnforcing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynOrderEnforcing")
            .field("has_reset", &self.has_reset)
            .field("env", &self.env)
            .finish()
    }
}

impl DynOrderEnforcing {
    pub(crate) fn new(env: Box<dyn DynEnv>) -> Self {
        Self {
            env,
            has_reset: false,
        }
    }
}

impl DynEnv for DynOrderEnforcing {
    fn step_dyn(&mut self, action: &DynValue) -> Result<StepResult<DynValue>> {
        if !self.has_reset {
            return Err(Error::ResetNeeded { method: "step" });
        }
        self.env.step_dyn(action)
    }

    fn reset_dyn(&mut self, seed: Option<u64>) -> Result<ResetResult<DynValue>> {
        self.has_reset = true;
        self.env.reset_dyn(seed)
    }

    fn render_dyn(&mut self) -> Result<RenderFrame> {
        if !self.has_reset {
            return Err(Error::ResetNeeded { method: "render" });
        }
        self.env.render_dyn()
    }

    fn close_dyn(&mut self) {
        self.env.close_dyn();
    }

    fn observation_space_info(&self) -> SpaceInfo {
        self.env.observation_space_info()
    }

    fn action_space_info(&self) -> SpaceInfo {
        self.env.action_space_info()
    }
}
