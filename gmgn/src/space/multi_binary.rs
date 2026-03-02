//! An n-dimensional binary space.
//!
//! Mirrors [Gymnasium `MultiBinary`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiBinary).

use rand::RngExt as _;

use crate::error::{Error, Result};
use crate::rng::Rng;
use crate::space::{Space, SpaceInfo};

/// A space of binary vectors of fixed length.
///
/// Each element is a `Vec<u8>` where every entry is either `0` or `1`.
/// Sampling draws each bit independently with equal probability.
///
/// # Examples
///
/// ```
/// use gmgn::space::{MultiBinary, Space};
/// use gmgn::rng::create_rng;
///
/// let space = MultiBinary::new(5).unwrap();
/// let mut rng = create_rng(Some(42));
/// let sample = space.sample(&mut rng);
/// assert_eq!(sample.len(), 5);
/// assert!(space.contains(&sample));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiBinary {
    /// Number of binary variables.
    n: usize,
    /// Cached shape `[n]`.
    shape: Vec<usize>,
}

impl MultiBinary {
    /// Create a new `MultiBinary` space with `n` binary variables.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpace`] if `n` is zero.
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(Error::InvalidSpace {
                reason: "n must be > 0".to_owned(),
            });
        }
        Ok(Self { n, shape: vec![n] })
    }

    /// The number of binary variables in this space.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }
}

impl Space for MultiBinary {
    type Element = Vec<u8>;

    fn sample(&self, rng: &mut Rng) -> Vec<u8> {
        (0..self.n).map(|_| rng.random_range(0..=1_u8)).collect()
    }

    fn contains(&self, value: &Vec<u8>) -> bool {
        value.len() == self.n && value.iter().all(|&v| v <= 1)
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn flatdim(&self) -> usize {
        self.n
    }

    fn space_info(&self) -> SpaceInfo {
        SpaceInfo::MultiBinary { n: self.n }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::create_rng;

    #[test]
    fn sample_is_binary() {
        let space = MultiBinary::new(10).unwrap();
        let mut rng = create_rng(Some(42));
        for _ in 0..100 {
            let s = space.sample(&mut rng);
            assert_eq!(s.len(), 10);
            assert!(s.iter().all(|&v| v <= 1));
        }
    }

    #[test]
    fn contains_validates() {
        let space = MultiBinary::new(3).unwrap();
        assert!(space.contains(&vec![0, 1, 0]));
        assert!(space.contains(&vec![1, 1, 1]));
        assert!(!space.contains(&vec![0, 2, 0]));
        assert!(!space.contains(&vec![0, 1]));
    }

    #[test]
    fn rejects_zero() {
        assert!(MultiBinary::new(0).is_err());
    }

    #[test]
    fn shape_and_flatdim() {
        let space = MultiBinary::new(5).unwrap();
        assert_eq!(space.shape(), &[5]);
        assert_eq!(space.flatdim(), 5);
    }
}
