//! Type-erased environment trait and blanket implementation.

use super::DynValue;
use crate::env::{Env, RenderFrame, ResetResult, StepResult};
use crate::error::{Error, Result};
use crate::space::{Space, SpaceInfo};

/// A type-erased environment that operates on [`DynValue`].
///
/// Every concrete `Env` whose observation/action types can convert to/from
/// [`DynValue`] automatically implements `DynEnv` via the blanket impl below.
pub trait DynEnv: std::fmt::Debug {
    /// Advance one timestep.
    ///
    /// # Errors
    ///
    /// Returns an error if the action type mismatches or the inner env fails.
    fn step_dyn(&mut self, action: &DynValue) -> Result<StepResult<DynValue>>;

    /// Reset to initial state.
    ///
    /// # Errors
    ///
    /// Returns an error if the inner environment fails to reset.
    fn reset_dyn(&mut self, seed: Option<u64>) -> Result<ResetResult<DynValue>>;

    /// Render a frame.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering is unsupported or the env is not reset.
    fn render_dyn(&mut self) -> Result<RenderFrame>;

    /// Clean up resources.
    fn close_dyn(&mut self);

    /// Runtime description of the observation space.
    fn observation_space_info(&self) -> SpaceInfo;

    /// Runtime description of the action space.
    fn action_space_info(&self) -> SpaceInfo;
}

impl<E> DynEnv for E
where
    E: Env + std::fmt::Debug,
    E::Obs: Into<DynValue>,
    E::Act: TryFrom<DynValue, Error = Error>,
    E::ObsSpace: Space,
    E::ActSpace: Space,
{
    fn step_dyn(&mut self, action: &DynValue) -> Result<StepResult<DynValue>> {
        let act = E::Act::try_from(action.clone())?;
        let r = self.step(&act)?;
        Ok(StepResult {
            obs: r.obs.into(),
            reward: r.reward,
            terminated: r.terminated,
            truncated: r.truncated,
            info: r.info,
        })
    }

    fn reset_dyn(&mut self, seed: Option<u64>) -> Result<ResetResult<DynValue>> {
        let r = self.reset(seed)?;
        Ok(ResetResult {
            obs: r.obs.into(),
            info: r.info,
        })
    }

    fn render_dyn(&mut self) -> Result<RenderFrame> {
        self.render()
    }

    fn close_dyn(&mut self) {
        self.close();
    }

    fn observation_space_info(&self) -> SpaceInfo {
        self.observation_space().space_info()
    }

    fn action_space_info(&self) -> SpaceInfo {
        self.action_space().space_info()
    }
}
