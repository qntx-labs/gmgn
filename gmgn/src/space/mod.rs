//! Observation and action space definitions.
//!
//! Spaces define the valid structure for observations and actions in an
//! environment. They support sampling random elements and membership testing.
//!
//! Mirrors [Gymnasium spaces](https://gymnasium.farama.org/api/spaces/) with
//! idiomatic Rust associated types.

mod bounded;
mod discrete;
mod multi_binary;
mod multi_discrete;

pub use bounded::BoundedSpace;
pub use discrete::Discrete;
pub use multi_binary::MultiBinary;
pub use multi_discrete::MultiDiscrete;

use crate::rng::Rng;

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
}
