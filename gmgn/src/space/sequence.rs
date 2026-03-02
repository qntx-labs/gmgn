//! Variable-length sequence space.
//!
//! Mirrors [Gymnasium `Sequence`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Sequence).
//!
//! Elements are `Vec<S::Element>` of varying length, where each item belongs
//! to a shared *feature space* `S`.
//!
//! # Examples
//!
//! ```
//! use gmgn::space::{SequenceSpace, BoundedSpace, Space};
//! use gmgn::rng::create_rng;
//!
//! let feature = BoundedSpace::new(vec![0.0], vec![1.0]).unwrap();
//! let space = SequenceSpace::new(feature, None, None);
//! let mut rng = create_rng(Some(42));
//! let sample = space.sample(&mut rng);
//! assert!(space.contains(&sample));
//! ```

use rand::RngExt as _;

use crate::rng::Rng;
use crate::space::{Space, SpaceInfo};

/// A space of variable-length sequences drawn from a single feature space.
///
/// Samples have length drawn from a geometric-like distribution (matching
/// Gymnasium behaviour) unless explicit bounds are set.
#[derive(Debug, Clone)]
pub struct SequenceSpace<S: Space> {
    /// The space that each element of the sequence belongs to.
    feature_space: S,
    /// Minimum sequence length (default: 1).
    min_len: usize,
    /// Maximum sequence length for sampling (default: 10).
    max_len: usize,
}

impl<S: Space> SequenceSpace<S> {
    /// Create a new sequence space.
    ///
    /// - `min_len`: minimum length (`None` defaults to 1).
    /// - `max_len`: maximum length for sampling (`None` defaults to 10).
    ///   `contains` does **not** enforce `max_len` — any length ≥ `min_len` is valid.
    ///
    /// # Panics
    ///
    /// Panics if `min_len > max_len` (after defaults are applied).
    #[must_use]
    pub fn new(feature_space: S, min_len: Option<usize>, max_len: Option<usize>) -> Self {
        let min_len = min_len.unwrap_or(1);
        let max_len = max_len.unwrap_or(10);
        assert!(
            min_len <= max_len,
            "min_len ({min_len}) must be <= max_len ({max_len})"
        );
        Self {
            feature_space,
            min_len,
            max_len,
        }
    }

    /// The feature space shared by all elements.
    #[must_use]
    pub const fn feature_space(&self) -> &S {
        &self.feature_space
    }

    /// The minimum sequence length.
    #[must_use]
    pub const fn min_len(&self) -> usize {
        self.min_len
    }

    /// The maximum sequence length used for sampling.
    #[must_use]
    pub const fn max_len(&self) -> usize {
        self.max_len
    }
}

impl<S: Space> Space for SequenceSpace<S> {
    type Element = Vec<S::Element>;

    fn sample(&self, rng: &mut Rng) -> Self::Element {
        let len = if self.min_len == self.max_len {
            self.min_len
        } else {
            rng.random_range(self.min_len..=self.max_len)
        };
        (0..len).map(|_| self.feature_space.sample(rng)).collect()
    }

    fn contains(&self, value: &Self::Element) -> bool {
        value.len() >= self.min_len && value.iter().all(|v| self.feature_space.contains(v))
    }

    fn shape(&self) -> &[usize] {
        // Variable-length — no fixed shape.
        &[]
    }

    fn flatdim(&self) -> usize {
        // Variable-length sequences cannot be meaningfully flattened to a
        // fixed-size vector. Return the feature space flatdim as a per-element
        // dimension hint.
        self.feature_space.flatdim()
    }

    fn is_flattenable(&self) -> bool {
        false
    }

    fn space_info(&self) -> SpaceInfo {
        SpaceInfo::Sequence(Box::new(self.feature_space.space_info()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::create_rng;
    use crate::space::{BoundedSpace, Discrete};

    #[test]
    fn sample_and_contains() {
        let space = SequenceSpace::new(Discrete::new(5), Some(2), Some(6));
        let mut rng = create_rng(Some(42));
        for _ in 0..30 {
            let sample = space.sample(&mut rng);
            assert!(space.contains(&sample), "sample {sample:?} not in space");
            assert!(sample.len() >= 2);
            assert!(sample.len() <= 6);
        }
    }

    #[test]
    fn bounded_feature_space() {
        let feature = BoundedSpace::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        let space = SequenceSpace::new(feature, None, Some(4));
        let mut rng = create_rng(Some(0));
        let sample = space.sample(&mut rng);
        assert!(space.contains(&sample));
        for elem in &sample {
            assert_eq!(elem.len(), 2);
        }
    }

    #[test]
    fn rejects_too_short() {
        let space = SequenceSpace::new(Discrete::new(3), Some(3), Some(5));
        assert!(!space.contains(&vec![0_i64, 1])); // len 2 < min 3
        assert!(space.contains(&vec![0, 1, 2])); // len 3 >= min 3
    }

    #[test]
    fn rejects_invalid_element() {
        let space = SequenceSpace::new(Discrete::new(3), Some(1), Some(5));
        assert!(!space.contains(&vec![5_i64])); // 5 not in {0,1,2}
    }

    #[test]
    fn not_flattenable() {
        let space = SequenceSpace::new(Discrete::new(3), None, None);
        assert!(!space.is_flattenable());
    }

    #[test]
    fn space_info_is_sequence() {
        let space = SequenceSpace::new(Discrete::new(3), None, None);
        assert!(matches!(space.space_info(), SpaceInfo::Sequence(_)));
    }

    #[test]
    #[should_panic(expected = "min_len")]
    fn inverted_bounds_panics() {
        let _ = SequenceSpace::new(Discrete::new(3), Some(10), Some(5));
    }
}
