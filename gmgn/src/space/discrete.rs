//! A finite space of integers `{start, start+1, ..., start+n-1}`.

use rand::RngExt as _;

use crate::rng::Rng;
use crate::space::Space;

/// A space consisting of `n` consecutive integers starting from `start`.
///
/// Represents the set `{start, start+1, ..., start+n-1}`.
///
/// # Examples
///
/// ```
/// use gmgn::space::{Discrete, Space};
/// use gmgn::rng::create_rng;
///
/// let space = Discrete::new(3); // {0, 1, 2}
/// let mut rng = create_rng(Some(42));
/// let sample = space.sample(&mut rng);
/// assert!(space.contains(&sample));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Discrete {
    /// The number of elements in this space.
    pub n: u64,
    /// The smallest element of this space.
    pub start: i64,
}

impl Discrete {
    /// Create a new `Discrete` space with `n` elements starting from 0.
    ///
    /// # Panics
    ///
    /// Panics if `n` is zero.
    #[must_use]
    pub fn new(n: u64) -> Self {
        assert!(n > 0, "Discrete space must have at least one element");
        Self { n, start: 0 }
    }

    /// Create a new `Discrete` space with `n` elements starting from `start`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is zero.
    #[must_use]
    pub fn with_start(n: u64, start: i64) -> Self {
        assert!(n > 0, "Discrete space must have at least one element");
        Self { n, start }
    }
}

impl Space for Discrete {
    type Element = i64;

    fn sample(&self, rng: &mut Rng) -> i64 {
        self.start + i64::from(rng.random_range(0..u32::try_from(self.n).expect("n fits u32")))
    }

    fn contains(&self, value: &i64) -> bool {
        *value >= self.start && *value < self.start + self.n.cast_signed()
    }

    fn shape(&self) -> &[usize] {
        // Scalar space — no dimensions.
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::create_rng;

    #[test]
    fn sample_is_within_bounds() {
        let space = Discrete::new(5);
        let mut rng = create_rng(Some(42));
        for _ in 0..100 {
            let s = space.sample(&mut rng);
            assert!(space.contains(&s), "sample {s} not in space");
        }
    }

    #[test]
    fn contains_checks_bounds() {
        let space = Discrete::with_start(3, -1); // {-1, 0, 1}
        assert!(space.contains(&-1));
        assert!(space.contains(&0));
        assert!(space.contains(&1));
        assert!(!space.contains(&-2));
        assert!(!space.contains(&2));
    }

    #[test]
    fn shape_is_empty() {
        let space = Discrete::new(2);
        assert!(space.shape().is_empty());
    }
}
