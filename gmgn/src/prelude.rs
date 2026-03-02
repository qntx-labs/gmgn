//! Convenience re-exports for common gmgn types.
//!
//! ```rust
//! use gmgn::prelude::*;
//! ```

pub use crate::env::{Env, Info, InfoValue, RenderFrame, RenderMode, ResetResult, StepResult};
pub use crate::error::{Error, Result};
pub use crate::space::{
    BoundedSpace, Discrete, MultiBinary, MultiDiscrete, Space, SpaceInfo, Tuple2, Tuple3,
};
