//! Observation and action space definitions.
//!
//! Spaces define the valid structure for observations and actions in an
//! environment. They support sampling random elements and membership testing.
//!
//! Mirrors [Gymnasium spaces](https://gymnasium.farama.org/api/spaces/) with
//! idiomatic Rust associated types.

mod bounded;
mod dict;
mod discrete;
mod multi_binary;
mod multi_discrete;
mod tuple;

pub use bounded::BoundedSpace;
pub use dict::{AnySpace, AnyValue, DictSpace};
pub use discrete::Discrete;
pub use multi_binary::MultiBinary;
pub use multi_discrete::MultiDiscrete;
pub use tuple::{Tuple2, Tuple3};

use crate::rng::Rng;

/// Describes the structure of a [`Space`] at runtime.
///
/// Used by [`DynEnv`](crate::registry::DynEnv) to expose observation and
/// action space metadata without requiring concrete generic types.
#[derive(Debug, Clone, PartialEq)]
pub enum SpaceInfo {
    /// A discrete space with `n` possible values `{0, 1, ..., n-1}`.
    Discrete {
        /// The number of discrete values.
        n: i64,
    },
    /// A bounded continuous space (box).
    Bounded {
        /// Lower bounds per dimension.
        low: Vec<f32>,
        /// Upper bounds per dimension.
        high: Vec<f32>,
        /// Shape of elements.
        shape: Vec<usize>,
    },
    /// A multi-discrete space with per-dimension counts.
    MultiDiscrete {
        /// Number of discrete values per dimension.
        nvec: Vec<i64>,
    },
    /// A multi-binary space with `n` binary elements.
    MultiBinary {
        /// Number of binary elements.
        n: usize,
    },
    /// A tuple (product) of sub-spaces.
    Tuple(Vec<Self>),
    /// A dictionary of named sub-spaces.
    Dict(Vec<(String, Self)>),
}

/// A space that defines the valid range for observations or actions.
///
/// Every space knows how to [`sample`](Space::sample) random elements and
/// [`contains`](Space::contains)-check membership. These two operations are the
/// minimum contract required by the reinforcement-learning loop.
pub trait Space {
    /// The concrete element type that this space produces and validates.
    type Element;

    /// Randomly sample a uniform element from this space.
    fn sample(&self, rng: &mut Rng) -> Self::Element;

    /// Check whether `value` is a valid member of this space.
    fn contains(&self, value: &Self::Element) -> bool;

    /// The shape of elements in this space as a slice of dimension sizes.
    ///
    /// Scalar spaces (e.g. [`Discrete`]) return an empty slice.
    fn shape(&self) -> &[usize];

    /// The total number of scalar values when this space is flattened.
    ///
    /// For [`Discrete`] this is `n`, for [`BoundedSpace`] it equals
    /// `shape().iter().product()`, etc.
    fn flatdim(&self) -> usize;

    /// Return a runtime description of this space's structure.
    fn space_info(&self) -> SpaceInfo;
}
