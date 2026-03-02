//! Dictionary space — a named collection of sub-spaces.
//!
//! Mirrors [Gymnasium `Dict`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Dict).
//!
//! Two flavours are provided:
//!
//! - **[`DictSpace<S>`]** — homogeneous dict where every sub-space has the
//!   same type `S`. This is the most common case (e.g. `GoalEnv` where every
//!   entry is a [`BoundedSpace`](super::BoundedSpace)).
//!
//! - **[`AnySpace`]** — an enum over all concrete space types, enabling
//!   heterogeneous dicts via `DictSpace<AnySpace>`.
//!
//! # Examples
//!
//! ```
//! use std::collections::HashMap;
//! use gmgn::space::{BoundedSpace, DictSpace, Space};
//! use gmgn::rng::create_rng;
//!
//! let space = DictSpace::new(vec![
//!     ("position".into(), BoundedSpace::new(vec![-1.0, -1.0], vec![1.0, 1.0]).unwrap()),
//!     ("velocity".into(), BoundedSpace::new(vec![-5.0], vec![5.0]).unwrap()),
//! ]);
//! let mut rng = create_rng(Some(42));
//! let sample = space.sample(&mut rng);
//! assert!(space.contains(&sample));
//! assert!(sample.contains_key("position"));
//! assert!(sample.contains_key("velocity"));
//! ```

use std::collections::HashMap;

use crate::rng::Rng;
use crate::space::{
    BoundedSpace, Discrete, MultiBinary, MultiDiscrete, Space, SpaceInfo, TextSpace,
};

/// A dictionary of named sub-spaces, all sharing the same type `S`.
///
/// Elements are `HashMap<String, S::Element>`.
#[derive(Debug, Clone)]
pub struct DictSpace<S: Space> {
    /// Ordered entries `(name, sub_space)`.
    entries: Vec<(String, S)>,
}

impl<S: Space> DictSpace<S> {
    /// Create a new dict space from named entries.
    ///
    /// Entries are stored in insertion order.
    #[must_use]
    pub const fn new(entries: Vec<(String, S)>) -> Self {
        Self { entries }
    }

    /// Number of entries in the dict.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the dict is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over `(name, sub_space)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &S)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Look up a sub-space by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&S> {
        self.entries.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }
}

impl<S: Space> Space for DictSpace<S> {
    type Element = HashMap<String, S::Element>;

    fn sample(&self, rng: &mut Rng) -> Self::Element {
        self.entries
            .iter()
            .map(|(k, s)| (k.clone(), s.sample(rng)))
            .collect()
    }

    fn contains(&self, value: &Self::Element) -> bool {
        if value.len() != self.entries.len() {
            return false;
        }
        self.entries
            .iter()
            .all(|(k, s)| value.get(k).is_some_and(|v| s.contains(v)))
    }

    fn shape(&self) -> &[usize] {
        // Composite spaces have no single shape.
        &[]
    }

    fn flatdim(&self) -> usize {
        self.entries.iter().map(|(_, s)| s.flatdim()).sum()
    }

    fn space_info(&self) -> SpaceInfo {
        SpaceInfo::Dict(
            self.entries
                .iter()
                .map(|(k, s)| (k.clone(), s.space_info()))
                .collect(),
        )
    }
}

/// A type-erased wrapper over concrete space types.
///
/// Enables heterogeneous [`DictSpace`] entries, e.g.:
///
/// ```rust,ignore
/// DictSpace::<AnySpace>::new(vec![
///     ("obs".into(), AnySpace::from(BoundedSpace::new(...))),
///     ("mode".into(), AnySpace::from(Discrete::new(3))),
/// ]);
/// ```
#[derive(Debug, Clone)]
pub enum AnySpace {
    /// A discrete space.
    Discrete(Discrete),
    /// A bounded continuous space.
    Bounded(BoundedSpace),
    /// A multi-discrete space.
    MultiDiscrete(MultiDiscrete),
    /// A multi-binary space.
    MultiBinary(MultiBinary),
    /// A text space.
    Text(TextSpace),
}

/// Dynamic value produced by [`AnySpace::sample`](Space::sample).
///
/// This is the element type for `DictSpace<AnySpace>` and can represent
/// any concrete observation or action value.
#[derive(Debug, Clone, PartialEq)]
pub enum AnyValue {
    /// From [`Discrete`].
    Discrete(i64),
    /// From [`BoundedSpace`].
    Continuous(Vec<f32>),
    /// From [`MultiDiscrete`].
    MultiDiscrete(Vec<i64>),
    /// From [`MultiBinary`].
    MultiBinary(Vec<u8>),
    /// From [`TextSpace`].
    Text(String),
}

impl Space for AnySpace {
    type Element = AnyValue;

    fn sample(&self, rng: &mut Rng) -> AnyValue {
        match self {
            Self::Discrete(s) => AnyValue::Discrete(s.sample(rng)),
            Self::Bounded(s) => AnyValue::Continuous(s.sample(rng)),
            Self::MultiDiscrete(s) => AnyValue::MultiDiscrete(s.sample(rng)),
            Self::MultiBinary(s) => AnyValue::MultiBinary(s.sample(rng)),
            Self::Text(s) => AnyValue::Text(s.sample(rng)),
        }
    }

    fn contains(&self, value: &AnyValue) -> bool {
        match (self, value) {
            (Self::Discrete(s), AnyValue::Discrete(v)) => s.contains(v),
            (Self::Bounded(s), AnyValue::Continuous(v)) => s.contains(v),
            (Self::MultiDiscrete(s), AnyValue::MultiDiscrete(v)) => s.contains(v),
            (Self::MultiBinary(s), AnyValue::MultiBinary(v)) => s.contains(v),
            (Self::Text(s), AnyValue::Text(v)) => s.contains(v),
            _ => false,
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::Discrete(s) => s.shape(),
            Self::Bounded(s) => s.shape(),
            Self::MultiDiscrete(s) => s.shape(),
            Self::MultiBinary(s) => s.shape(),
            Self::Text(s) => s.shape(),
        }
    }

    fn flatdim(&self) -> usize {
        match self {
            Self::Discrete(s) => s.flatdim(),
            Self::Bounded(s) => s.flatdim(),
            Self::MultiDiscrete(s) => s.flatdim(),
            Self::MultiBinary(s) => s.flatdim(),
            Self::Text(s) => s.flatdim(),
        }
    }

    fn space_info(&self) -> SpaceInfo {
        match self {
            Self::Discrete(s) => s.space_info(),
            Self::Bounded(s) => s.space_info(),
            Self::MultiDiscrete(s) => s.space_info(),
            Self::MultiBinary(s) => s.space_info(),
            Self::Text(s) => s.space_info(),
        }
    }
}

impl From<Discrete> for AnySpace {
    fn from(s: Discrete) -> Self {
        Self::Discrete(s)
    }
}

impl From<BoundedSpace> for AnySpace {
    fn from(s: BoundedSpace) -> Self {
        Self::Bounded(s)
    }
}

impl From<MultiDiscrete> for AnySpace {
    fn from(s: MultiDiscrete) -> Self {
        Self::MultiDiscrete(s)
    }
}

impl From<MultiBinary> for AnySpace {
    fn from(s: MultiBinary) -> Self {
        Self::MultiBinary(s)
    }
}

impl From<TextSpace> for AnySpace {
    fn from(s: TextSpace) -> Self {
        Self::Text(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::create_rng;

    #[test]
    fn homogeneous_dict_sample_and_contains() {
        let space = DictSpace::new(vec![
            (
                "position".into(),
                BoundedSpace::new(vec![-1.0, -1.0], vec![1.0, 1.0]).unwrap(),
            ),
            (
                "velocity".into(),
                BoundedSpace::new(vec![-5.0], vec![5.0]).unwrap(),
            ),
        ]);
        let mut rng = create_rng(Some(42));
        let sample = space.sample(&mut rng);
        assert!(space.contains(&sample));
        assert_eq!(sample.len(), 2);
        assert_eq!(sample["position"].len(), 2);
        assert_eq!(sample["velocity"].len(), 1);
    }

    #[test]
    fn heterogeneous_dict_sample_and_contains() {
        let space = DictSpace::<AnySpace>::new(vec![
            (
                "obs".into(),
                BoundedSpace::new(vec![-1.0], vec![1.0]).unwrap().into(),
            ),
            ("mode".into(), Discrete::new(3).into()),
        ]);
        let mut rng = create_rng(Some(42));
        let sample = space.sample(&mut rng);
        assert!(space.contains(&sample));
        assert_eq!(sample.len(), 2);
    }

    #[test]
    fn rejects_missing_key() {
        let space = DictSpace::new(vec![
            ("a".into(), Discrete::new(2)),
            ("b".into(), Discrete::new(3)),
        ]);
        let mut rng = create_rng(Some(42));
        let mut sample = space.sample(&mut rng);
        sample.remove("b");
        assert!(!space.contains(&sample));
    }

    #[test]
    fn flatdim_sums_entries() {
        let space = DictSpace::new(vec![
            (
                "pos".into(),
                BoundedSpace::new(vec![0.0; 3], vec![1.0; 3]).unwrap(),
            ),
            (
                "vel".into(),
                BoundedSpace::new(vec![0.0; 2], vec![1.0; 2]).unwrap(),
            ),
        ]);
        assert_eq!(space.flatdim(), 5);
    }

    #[test]
    fn space_info_is_dict() {
        let space = DictSpace::new(vec![("x".into(), Discrete::new(4))]);
        let info = space.space_info();
        assert!(matches!(info, SpaceInfo::Dict(_)));
    }

    #[test]
    fn get_and_len() {
        let space = DictSpace::new(vec![
            ("a".into(), Discrete::new(2)),
            ("b".into(), Discrete::new(5)),
        ]);
        assert_eq!(space.len(), 2);
        assert!(!space.is_empty());
        assert!(space.get("a").is_some());
        assert!(space.get("c").is_none());
    }
}
