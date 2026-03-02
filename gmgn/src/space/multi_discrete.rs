//! Cartesian product of multiple [`Discrete`](super::Discrete) spaces.
//!
//! Mirrors [Gymnasium `MultiDiscrete`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete).

use rand::RngExt as _;

use crate::error::{Error, Result};
use crate::rng::Rng;
use crate::space::{Space, SpaceInfo};

/// The cartesian product of multiple [`Discrete`](super::Discrete) spaces.
///
/// Each dimension `i` takes integer values in `[start[i], start[i] + nvec[i])`.
///
/// Useful for representing game controllers, multi-button inputs, or any
/// action/observation that consists of several independent categorical variables.
///
/// # Examples
///
/// ```
/// use gmgn::space::{MultiDiscrete, Space};
/// use gmgn::rng::create_rng;
///
/// // Three discrete dimensions: {0..5}, {0..2}, {0..2}
/// let space = MultiDiscrete::new(vec![5, 2, 2]).unwrap();
/// let mut rng = create_rng(Some(42));
/// let sample = space.sample(&mut rng);
/// assert!(space.contains(&sample));
/// assert_eq!(space.shape(), &[3]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiDiscrete {
    /// Number of values each categorical variable can take.
    nvec: Vec<u64>,
    /// Starting value for each categorical variable.
    start: Vec<i64>,
    /// Cached shape derived from `nvec.len()`.
    shape: Vec<usize>,
}

impl MultiDiscrete {
    /// Create a new `MultiDiscrete` space with each dimension starting from 0.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpace`] if `nvec` is empty or any element is zero.
    pub fn new(nvec: Vec<u64>) -> Result<Self> {
        Self::with_start(nvec, None)
    }

    /// Create a new `MultiDiscrete` space with explicit starting values.
    ///
    /// If `start` is `None`, all dimensions start from 0.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpace`] if `nvec` is empty, any element is zero,
    /// or `start` length does not match `nvec`.
    pub fn with_start(nvec: Vec<u64>, start: Option<Vec<i64>>) -> Result<Self> {
        if nvec.is_empty() {
            return Err(Error::InvalidSpace {
                reason: "nvec must not be empty".to_owned(),
            });
        }
        if nvec.contains(&0) {
            return Err(Error::InvalidSpace {
                reason: "all nvec elements must be > 0".to_owned(),
            });
        }

        let start = match start {
            Some(s) => {
                if s.len() != nvec.len() {
                    return Err(Error::InvalidSpace {
                        reason: format!(
                            "start length ({}) must match nvec length ({})",
                            s.len(),
                            nvec.len()
                        ),
                    });
                }
                s
            }
            None => vec![0; nvec.len()],
        };

        let shape = vec![nvec.len()];
        Ok(Self { nvec, start, shape })
    }

    /// The number of values each categorical variable can take.
    #[must_use]
    pub fn nvec(&self) -> &[u64] {
        &self.nvec
    }

    /// The starting value for each categorical variable.
    #[must_use]
    pub fn start(&self) -> &[i64] {
        &self.start
    }
}

impl Space for MultiDiscrete {
    type Element = Vec<i64>;

    fn sample(&self, rng: &mut Rng) -> Vec<i64> {
        self.nvec
            .iter()
            .zip(self.start.iter())
            .map(|(&n, &s)| {
                s + i64::from(rng.random_range(0..u32::try_from(n).expect("n fits u32")))
            })
            .collect()
    }

    fn contains(&self, value: &Vec<i64>) -> bool {
        if value.len() != self.nvec.len() {
            return false;
        }
        value
            .iter()
            .zip(self.nvec.iter().zip(self.start.iter()))
            .all(|(&v, (&n, &s))| v >= s && v < s + n.cast_signed())
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[allow(clippy::cast_possible_truncation)] // nvec elements are small in practice.
    fn flatdim(&self) -> usize {
        self.nvec.iter().map(|&n| n as usize).sum()
    }

    fn space_info(&self) -> SpaceInfo {
        SpaceInfo::MultiDiscrete {
            nvec: self.nvec.iter().map(|&n| n.cast_signed()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::create_rng;

    #[test]
    fn sample_within_bounds() {
        let space = MultiDiscrete::new(vec![5, 2, 2]).unwrap();
        let mut rng = create_rng(Some(42));
        for _ in 0..100 {
            let s = space.sample(&mut rng);
            assert!(space.contains(&s), "sample {s:?} not in space");
        }
    }

    #[test]
    fn contains_with_start() {
        let space = MultiDiscrete::with_start(vec![3, 2], Some(vec![-1, 5])).unwrap();
        assert!(space.contains(&vec![-1, 5]));
        assert!(space.contains(&vec![1, 6]));
        assert!(!space.contains(&vec![-2, 5]));
        assert!(!space.contains(&vec![2, 5]));
        assert!(!space.contains(&vec![0, 7]));
    }

    #[test]
    fn rejects_empty_nvec() {
        assert!(MultiDiscrete::new(vec![]).is_err());
    }

    #[test]
    fn rejects_zero_element() {
        assert!(MultiDiscrete::new(vec![3, 0, 2]).is_err());
    }

    #[test]
    fn rejects_mismatched_start() {
        assert!(MultiDiscrete::with_start(vec![3, 2], Some(vec![0])).is_err());
    }

    #[test]
    fn shape_equals_ndims() {
        let space = MultiDiscrete::new(vec![5, 2, 2]).unwrap();
        assert_eq!(space.shape(), &[3]);
    }

    #[test]
    fn flatdim_is_sum_of_nvec() {
        let space = MultiDiscrete::new(vec![5, 2, 2]).unwrap();
        assert_eq!(space.flatdim(), 9);
    }
}
