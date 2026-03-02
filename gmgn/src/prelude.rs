//! Convenience re-exports for common gmgn types.
//!
//! ```rust
//! use gmgn::prelude::*;
//! ```

pub use crate::env::{Env, RenderFrame, RenderMode, ResetResult, StepResult};
pub use crate::error::{Error, Result};
pub use crate::space::{BoundedSpace, Discrete, Space};
pub use crate::wrapper::Wrapper;
