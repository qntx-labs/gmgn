//! Core environment trait and associated types.
//!
//! The [`Env`] trait is the central abstraction in gmgn, defining how an
//! agent interacts with a reinforcement learning environment through
//! [`step`](Env::step) and [`reset`](Env::reset).
//!
//! Mirrors [Gymnasium `Env`](https://gymnasium.farama.org/api/env/) adapted to
//! idiomatic Rust with associated types.

use std::collections::HashMap;

use crate::error::Result;
use crate::space::Space;

/// A single value in the auxiliary [`Info`] dictionary.
///
/// Covers the most common payload types that Gymnasium environments return
/// while remaining `Debug + Clone + Send` without requiring `serde` or `Any`.
#[derive(Debug, Clone, PartialEq)]
pub enum InfoValue {
    /// Boolean flag (e.g. `TimeLimit.truncated`).
    Bool(bool),
    /// Integer metric (e.g. episode length).
    Int(i64),
    /// Floating-point metric (e.g. episode reward).
    Float(f64),
    /// Free-form string.
    String(String),
    /// Array of integers (e.g. action masks).
    IntArray(Vec<i64>),
}

/// Auxiliary diagnostic information returned alongside observations.
///
/// Mirrors the `info` dict in Gymnasium. Used for debugging, logging, and
/// passing environment-specific metadata such as episode statistics.
pub type Info = HashMap<String, InfoValue>;

/// The result of a single environment [`step`](Env::step).
#[derive(Debug, Clone)]
pub struct StepResult<O> {
    /// The observation after taking the action.
    pub obs: O,
    /// The scalar reward signal.
    pub reward: f64,
    /// Whether the agent reached a terminal state (MDP termination).
    pub terminated: bool,
    /// Whether the episode was cut short by an external condition (e.g. time limit).
    pub truncated: bool,
    /// Auxiliary diagnostic information.
    pub info: Info,
}

/// The result of an environment [`reset`](Env::reset).
#[derive(Debug, Clone)]
pub struct ResetResult<O> {
    /// The initial observation of the new episode.
    pub obs: O,
    /// Auxiliary diagnostic information.
    pub info: Info,
}

/// How the environment should be rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderMode {
    /// No rendering (fastest, for training).
    #[default]
    None,
    /// Return an RGB pixel buffer for programmatic consumption.
    RgbArray,
    /// Display a window for human observation.
    Human,
    /// Return an ANSI text representation.
    Ansi,
}

/// A rendered frame produced by [`Env::render`].
#[derive(Debug, Clone)]
pub enum RenderFrame {
    /// No frame produced.
    None,
    /// ANSI text representation.
    Ansi(String),
    /// RGB pixel buffer with dimensions.
    RgbArray {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Raw pixel data in RGB format (3 bytes per pixel).
        data: Vec<u8>,
    },
}

/// Static metadata about an environment's capabilities.
///
/// Mirrors Gymnasium's `metadata` dict (e.g. `render_modes`, `render_fps`).
#[derive(Debug, Clone, Copy)]
pub struct EnvMetadata {
    /// Supported render modes (e.g. `["human", "rgb_array"]`).
    pub render_modes: &'static [&'static str],
    /// Target frames per second for rendering (`None` = unspecified).
    pub render_fps: Option<u32>,
}

impl EnvMetadata {
    /// Default metadata for environments that do not specify capabilities.
    pub const DEFAULT: Self = Self {
        render_modes: &[],
        render_fps: None,
    };
}

/// A reinforcement learning environment.
///
/// This is the core trait that all environments must implement. It mirrors
/// the Gymnasium `Env` API adapted to idiomatic Rust with associated types.
///
/// # Associated Types
///
/// - `Obs` — The observation type returned by `step` and `reset`.
/// - `Act` — The action type accepted by `step`.
/// - `ObsSpace` — The space that describes valid observations.
/// - `ActSpace` — The space that describes valid actions.
///
/// # Lifecycle
///
/// 1. Create the environment via its constructor (or [`make`](crate::registry::make)).
/// 2. Call [`reset`](Env::reset) to obtain the first observation.
/// 3. Repeatedly call [`step`](Env::step) with actions.
/// 4. Call [`close`](Env::close) when done.
pub trait Env {
    /// The observation type produced by this environment.
    type Obs;
    /// The action type consumed by this environment.
    type Act;
    /// The observation space type.
    type ObsSpace: Space<Element = Self::Obs>;
    /// The action space type.
    type ActSpace: Space<Element = Self::Act>;

    /// Advance the environment by one timestep with the given action.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment has not been reset, or if the
    /// action is invalid.
    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>>;

    /// Reset the environment to an initial state.
    ///
    /// If `seed` is provided, the environment's RNG is re-seeded for
    /// reproducibility. Subsequent calls with `seed = None` do not
    /// alter the RNG.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment cannot be initialized.
    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>>;

    /// Compute a render frame based on the current state.
    ///
    /// The behavior depends on the [`RenderMode`] set during construction.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering is not supported or the environment
    /// has not been reset.
    fn render(&mut self) -> Result<RenderFrame>;

    /// Clean up resources held by the environment.
    ///
    /// Calling `close` on an already-closed environment is a no-op.
    fn close(&mut self) {}

    /// The observation space describing valid observations.
    fn observation_space(&self) -> &Self::ObsSpace;

    /// The action space describing valid actions.
    fn action_space(&self) -> &Self::ActSpace;

    /// The render mode configured for this environment.
    fn render_mode(&self) -> &RenderMode;

    /// Static metadata about this environment's capabilities.
    fn metadata(&self) -> EnvMetadata {
        EnvMetadata::DEFAULT
    }
}
