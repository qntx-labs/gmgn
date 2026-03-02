//! Observation and action space definitions.
//!
//! Spaces define the valid structure for observations and actions in an
//! environment. They support sampling random elements and membership testing.

mod bounded;
mod discrete;

pub use bounded::BoundedSpace;
pub use discrete::Discrete;

use crate::rng::Rng;

/// A space that defines the valid range for observations or actions.
pub trait Space {
    /// The concrete element type that this space produces and validates.
    type Element;

    /// Randomly sample an element from this space.
    fn sample(&self, rng: &mut Rng) -> Self::Element;

    /// Check whether a value is a valid member of this space.
    fn contains(&self, value: &Self::Element) -> bool;

    /// The shape of elements in this space as a slice of dimension sizes.
    ///
    /// Scalar spaces (e.g. [`Discrete`]) return an empty slice.
    fn shape(&self) -> &[usize];
}
