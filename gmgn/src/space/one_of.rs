//! Union (direct sum) of multiple spaces — exactly one is active per sample.
//!
//! Mirrors [Gymnasium `OneOf`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.OneOf).
//!
//! Elements are `(index, value)` pairs where `index` selects which sub-space
//! produced the `value`.
//!
//! # Examples
//!
//! ```
//! use gmgn::space::{OneOf, AnySpace, AnyValue, BoundedSpace, Discrete, Space};
//! use gmgn::rng::create_rng;
//!
//! let space = OneOf::new(vec![
//!     AnySpace::from(Discrete::new(3)),
//!     AnySpace::from(BoundedSpace::new(vec![-1.0], vec![1.0]).unwrap()),
//! ]);
//! let mut rng = create_rng(Some(42));
//! let (idx, val) = space.sample(&mut rng);
//! assert!(space.contains(&(idx, val)));
//! ```

use rand::RngExt as _;

use crate::rng::Rng;
use crate::space::dict::{AnySpace, AnyValue};
use crate::space::{Space, SpaceInfo};

/// A union of sub-spaces where each sample comes from exactly one of them.
///
/// The element type is `(usize, AnyValue)` — a pair of the selected
/// sub-space index and the sampled value from that sub-space.
#[derive(Debug, Clone)]
pub struct OneOf {
    /// The constituent sub-spaces.
    spaces: Vec<AnySpace>,
}

impl OneOf {
    /// Create a new `OneOf` space from a list of sub-spaces.
    ///
    /// # Panics
    ///
    /// Panics if `spaces` is empty.
    #[must_use]
    pub fn new(spaces: Vec<AnySpace>) -> Self {
        assert!(!spaces.is_empty(), "OneOf requires at least one sub-space");
        Self { spaces }
    }

    /// The number of sub-spaces.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.spaces.len()
    }

    /// Whether the space has no sub-spaces (always `false` after construction).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.spaces.is_empty()
    }

    /// Borrow the sub-space at `index`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&AnySpace> {
        self.spaces.get(index)
    }

    /// Iterate over all sub-spaces.
    pub fn iter(&self) -> impl Iterator<Item = &AnySpace> {
        self.spaces.iter()
    }
}

impl Space for OneOf {
    type Element = (usize, AnyValue);

    fn sample(&self, rng: &mut Rng) -> Self::Element {
        let idx = rng.random_range(0..self.spaces.len());
        let val = self.spaces[idx].sample(rng);
        (idx, val)
    }

    fn contains(&self, value: &Self::Element) -> bool {
        let (idx, val) = value;
        self.spaces
            .get(*idx)
            .is_some_and(|space| space.contains(val))
    }

    fn shape(&self) -> &[usize] {
        &[]
    }

    fn flatdim(&self) -> usize {
        // OneOf flatdim = 1 (for index) + max flatdim across sub-spaces.
        1 + self.spaces.iter().map(Space::flatdim).max().unwrap_or(0)
    }

    fn is_flattenable(&self) -> bool {
        self.spaces.iter().all(Space::is_flattenable)
    }

    fn space_info(&self) -> SpaceInfo {
        SpaceInfo::OneOf(self.spaces.iter().map(Space::space_info).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::create_rng;
    use crate::space::{BoundedSpace, Discrete};

    #[test]
    fn sample_and_contains() {
        let space = OneOf::new(vec![
            AnySpace::from(Discrete::new(5)),
            AnySpace::from(BoundedSpace::new(vec![-1.0], vec![1.0]).unwrap()),
        ]);
        let mut rng = create_rng(Some(42));
        for _ in 0..50 {
            let sample = space.sample(&mut rng);
            assert!(space.contains(&sample), "sample {sample:?} not in space");
            assert!(sample.0 < 2);
        }
    }

    #[test]
    fn rejects_wrong_index() {
        let space = OneOf::new(vec![AnySpace::from(Discrete::new(3))]);
        assert!(!space.contains(&(1, AnyValue::Discrete(0))));
    }

    #[test]
    fn rejects_wrong_type() {
        let space = OneOf::new(vec![AnySpace::from(Discrete::new(3))]);
        assert!(!space.contains(&(0, AnyValue::Continuous(vec![1.0]))));
    }

    #[test]
    fn len_and_get() {
        let space = OneOf::new(vec![
            AnySpace::from(Discrete::new(2)),
            AnySpace::from(Discrete::new(4)),
        ]);
        assert_eq!(space.len(), 2);
        assert!(space.get(0).is_some());
        assert!(space.get(2).is_none());
    }

    #[test]
    fn space_info_is_one_of() {
        let space = OneOf::new(vec![AnySpace::from(Discrete::new(3))]);
        assert!(matches!(space.space_info(), SpaceInfo::OneOf(_)));
    }

    #[test]
    #[should_panic(expected = "at least one")]
    fn empty_panics() {
        let _ = OneOf::new(vec![]);
    }
}
