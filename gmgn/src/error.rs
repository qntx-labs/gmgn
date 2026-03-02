//! Error types for the gmgn reinforcement learning library.
//!
//! All fallible operations return [`Result<T>`] which uses [`Error`] as the
//! error type.

use std::result;

/// All errors that can occur within the gmgn library.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The environment has not been reset before calling `step` or `render`.
    #[error("environment must be reset before calling `{method}`")]
    ResetNeeded {
        /// The method that was called without a prior reset.
        method: &'static str,
    },

    /// An invalid action was provided to the environment.
    #[error("invalid action: {reason}")]
    InvalidAction {
        /// Description of why the action is invalid.
        reason: String,
    },

    /// The requested render mode is not supported by this environment.
    #[error("unsupported render mode: {mode}")]
    UnsupportedRenderMode {
        /// The render mode that was requested.
        mode: String,
    },

    /// A required optional dependency is not available.
    #[error("missing dependency: {0}")]
    MissingDependency(String),

    /// An invalid space configuration was provided.
    #[error("invalid space: {reason}")]
    InvalidSpace {
        /// Description of why the space configuration is invalid.
        reason: String,
    },

    /// No environment registered under the given id.
    #[error("environment not found: `{id}`")]
    NotRegistered {
        /// The id that was looked up.
        id: String,
    },

    /// An environment id has already been registered.
    #[error("environment already registered: `{id}`")]
    AlreadyRegistered {
        /// The duplicate id.
        id: String,
    },

    /// Type conversion failed in dynamic dispatch.
    #[error("type mismatch: {reason}")]
    TypeMismatch {
        /// Description of the mismatch.
        reason: String,
    },
}

/// A specialized `Result` type for gmgn operations.
pub type Result<T> = result::Result<T, Error>;
