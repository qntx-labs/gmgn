//! A continuous box in ℝⁿ defined by per-element lower and upper bounds.

use rand::RngExt as _;
use rand_distr::{Exp, StandardNormal};

use crate::error::{Error, Result};
use crate::rng::Rng;
use crate::space::{Space, SpaceInfo};

/// A continuous space representing the Cartesian product of closed intervals.
///
/// Each dimension `i` is bounded by `[low[i], high[i]]`. Supports unbounded
/// dimensions via `f32::NEG_INFINITY` / `f32::INFINITY`.
///
/// This is the Rust equivalent of Gymnasium's `Box` space, renamed to
/// [`BoundedSpace`] to avoid confusion with `std::boxed::Box`.
///
/// # Examples
///
/// ```
/// use gmgn::space::{BoundedSpace, Space};
/// use gmgn::rng::create_rng;
///
/// let space = BoundedSpace::new(
///     vec![-1.0, -2.0],
///     vec![1.0, 2.0],
/// ).unwrap();
/// let mut rng = create_rng(Some(42));
/// let sample = space.sample(&mut rng);
/// assert!(space.contains(&sample));
/// assert_eq!(space.shape(), &[2]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedSpace {
    /// Lower bounds for each dimension.
    pub low: Vec<f32>,
    /// Upper bounds for each dimension.
    pub high: Vec<f32>,
    /// Shape of the space (derived from bounds length).
    shape: Vec<usize>,
}

impl BoundedSpace {
    /// Create a new bounded space from per-element lower and upper bounds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpace`] if `low` and `high` have different
    /// lengths, or if any `low[i] > high[i]` for finite bounds.
    pub fn new(low: Vec<f32>, high: Vec<f32>) -> Result<Self> {
        if low.len() != high.len() {
            return Err(Error::InvalidSpace {
                reason: format!(
                    "low and high must have the same length, got {} and {}",
                    low.len(),
                    high.len()
                ),
            });
        }

        for (i, (l, h)) in low.iter().zip(high.iter()).enumerate() {
            if l.is_finite() && h.is_finite() && l > h {
                return Err(Error::InvalidSpace {
                    reason: format!("low[{i}] ({l}) > high[{i}] ({h})"),
                });
            }
        }

        let shape = vec![low.len()];
        Ok(Self { low, high, shape })
    }

    /// Create a bounded space where all dimensions share the same bounds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpace`] if `low > high` (when both are finite)
    /// or if `size` is zero.
    pub fn uniform(low: f32, high: f32, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(Error::InvalidSpace {
                reason: "size must be > 0".to_owned(),
            });
        }
        Self::new(vec![low; size], vec![high; size])
    }
}

impl Space for BoundedSpace {
    type Element = Vec<f32>;

    fn sample(&self, rng: &mut Rng) -> Vec<f32> {
        // Matches Gymnasium Box.sample() distribution per interval type:
        //   [a, b]      → Uniform
        //   [a, +∞)     → shifted Exponential
        //   (−∞, b]     → shifted negative Exponential
        //   (−∞, +∞)    → Standard Normal
        let exp = Exp::new(1.0_f32).expect("lambda=1 is valid");
        self.low
            .iter()
            .zip(self.high.iter())
            .map(|(&lo, &hi)| {
                if lo.is_finite() && hi.is_finite() {
                    rng.random_range(lo..=hi)
                } else if lo.is_finite() {
                    // [lo, +inf): shifted exponential distribution
                    lo + rng.sample(exp)
                } else if hi.is_finite() {
                    // (-inf, hi]: shifted negative exponential distribution
                    hi - rng.sample(exp)
                } else {
                    // (-inf, +inf): standard normal distribution
                    rng.sample::<f32, _>(StandardNormal)
                }
            })
            .collect()
    }

    fn contains(&self, value: &Vec<f32>) -> bool {
        if value.len() != self.low.len() {
            return false;
        }
        value
            .iter()
            .zip(self.low.iter().zip(self.high.iter()))
            .all(|(&v, (&lo, &hi))| v >= lo && v <= hi)
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn flatdim(&self) -> usize {
        self.low.len()
    }

    fn space_info(&self) -> SpaceInfo {
        SpaceInfo::Bounded {
            low: self.low.clone(),
            high: self.high.clone(),
            shape: self.shape.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::create_rng;

    #[test]
    fn sample_within_bounds() {
        let space = BoundedSpace::new(vec![-1.0, -2.0], vec![1.0, 2.0]).unwrap();
        let mut rng = create_rng(Some(42));
        for _ in 0..100 {
            let s = space.sample(&mut rng);
            assert!(space.contains(&s), "sample {s:?} not in space");
        }
    }

    #[test]
    fn contains_validates_bounds() {
        let space = BoundedSpace::new(vec![0.0], vec![1.0]).unwrap();
        assert!(space.contains(&vec![0.5]));
        assert!(space.contains(&vec![0.0]));
        assert!(space.contains(&vec![1.0]));
        assert!(!space.contains(&vec![-0.1]));
        assert!(!space.contains(&vec![1.1]));
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let result = BoundedSpace::new(vec![0.0, 0.0], vec![1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_inverted_bounds() {
        let result = BoundedSpace::new(vec![1.0], vec![0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn uniform_constructor() {
        let space = BoundedSpace::uniform(-1.0, 1.0, 4).unwrap();
        assert_eq!(space.shape(), &[4]);
        assert_eq!(space.low, vec![-1.0; 4]);
    }
}
