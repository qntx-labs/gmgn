//! Heterogeneous tuple spaces — cartesian products of different space types.
//!
//! Mirrors [Gymnasium `Tuple`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Tuple)
//! using Rust generics for zero-cost, type-safe composition.
//!
//! # Examples
//!
//! ```
//! use gmgn::space::{Discrete, Tuple3, Space};
//! use gmgn::rng::create_rng;
//!
//! // Blackjack-style observation: (player_sum, dealer_card, usable_ace)
//! let space = Tuple3::new(
//!     Discrete::new(32),
//!     Discrete::new(11),
//!     Discrete::new(2),
//! );
//! let mut rng = create_rng(Some(42));
//! let sample = space.sample(&mut rng);
//! assert!(space.contains(&sample));
//! ```

use crate::rng::Rng;
use crate::space::{Space, SpaceInfo};

/// Cartesian product of two spaces.
///
/// `Element = (A::Element, B::Element)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tuple2<A: Space, B: Space> {
    /// First sub-space.
    pub first: A,
    /// Second sub-space.
    pub second: B,
}

impl<A: Space, B: Space> Tuple2<A, B> {
    /// Create a new 2-tuple space.
    #[must_use]
    pub const fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A: Space, B: Space> Space for Tuple2<A, B> {
    type Element = (A::Element, B::Element);

    fn sample(&self, rng: &mut Rng) -> Self::Element {
        (self.first.sample(rng), self.second.sample(rng))
    }

    fn contains(&self, value: &Self::Element) -> bool {
        self.first.contains(&value.0) && self.second.contains(&value.1)
    }

    fn shape(&self) -> &[usize] {
        // Composite spaces have no single shape; return empty.
        &[]
    }

    fn flatdim(&self) -> usize {
        self.first.flatdim() + self.second.flatdim()
    }

    fn space_info(&self) -> SpaceInfo {
        SpaceInfo::Tuple(vec![self.first.space_info(), self.second.space_info()])
    }
}

/// Cartesian product of three spaces.
///
/// `Element = (A::Element, B::Element, C::Element)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tuple3<A: Space, B: Space, C: Space> {
    /// First sub-space.
    pub first: A,
    /// Second sub-space.
    pub second: B,
    /// Third sub-space.
    pub third: C,
}

impl<A: Space, B: Space, C: Space> Tuple3<A, B, C> {
    /// Create a new 3-tuple space.
    #[must_use]
    pub const fn new(first: A, second: B, third: C) -> Self {
        Self {
            first,
            second,
            third,
        }
    }
}

impl<A: Space, B: Space, C: Space> Space for Tuple3<A, B, C> {
    type Element = (A::Element, B::Element, C::Element);

    fn sample(&self, rng: &mut Rng) -> Self::Element {
        (
            self.first.sample(rng),
            self.second.sample(rng),
            self.third.sample(rng),
        )
    }

    fn contains(&self, value: &Self::Element) -> bool {
        self.first.contains(&value.0)
            && self.second.contains(&value.1)
            && self.third.contains(&value.2)
    }

    fn shape(&self) -> &[usize] {
        &[]
    }

    fn flatdim(&self) -> usize {
        self.first.flatdim() + self.second.flatdim() + self.third.flatdim()
    }

    fn space_info(&self) -> SpaceInfo {
        SpaceInfo::Tuple(vec![
            self.first.space_info(),
            self.second.space_info(),
            self.third.space_info(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::create_rng;
    use crate::space::Discrete;

    #[test]
    fn tuple2_sample_and_contains() {
        let space = Tuple2::new(Discrete::new(5), Discrete::new(3));
        let mut rng = create_rng(Some(42));
        for _ in 0..100 {
            let s = space.sample(&mut rng);
            assert!(space.contains(&s), "sample {s:?} not in space");
        }
    }

    #[test]
    fn tuple2_rejects_invalid() {
        let space = Tuple2::new(Discrete::new(2), Discrete::new(3));
        assert!(!space.contains(&(2, 0)));
        assert!(!space.contains(&(0, 3)));
        assert!(space.contains(&(1, 2)));
    }

    #[test]
    fn tuple3_sample_and_contains() {
        let space = Tuple3::new(Discrete::new(32), Discrete::new(11), Discrete::new(2));
        let mut rng = create_rng(Some(99));
        for _ in 0..100 {
            let s = space.sample(&mut rng);
            assert!(space.contains(&s), "sample {s:?} not in space");
        }
    }

    #[test]
    fn tuple3_flatdim() {
        let space = Tuple3::new(Discrete::new(32), Discrete::new(11), Discrete::new(2));
        assert_eq!(space.flatdim(), 32 + 11 + 2);
    }

    #[test]
    fn tuple2_flatdim() {
        let space = Tuple2::new(Discrete::new(10), Discrete::new(5));
        assert_eq!(space.flatdim(), 15);
    }

    #[test]
    fn shape_is_empty_for_composite() {
        let s2 = Tuple2::new(Discrete::new(3), Discrete::new(4));
        assert!(s2.shape().is_empty());
        let s3 = Tuple3::new(Discrete::new(1), Discrete::new(2), Discrete::new(3));
        assert!(s3.shape().is_empty());
    }
}
