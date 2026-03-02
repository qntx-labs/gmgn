//! Flatten and unflatten space elements for neural network input.
//!
//! Mirrors [Gymnasium `spaces.utils.flatten`](https://gymnasium.farama.org/api/spaces/utils/)
//! providing conversions between structured observations and flat `Vec<f32>`
//! vectors suitable for policy networks.
//!
//! # Examples
//!
//! ```
//! use gmgn::space::{Discrete, Space};
//! use gmgn::space::flatten::{flatten_discrete, unflatten_discrete};
//!
//! let space = Discrete::new(4);
//! let flat = flatten_discrete(&space, 2);
//! assert_eq!(flat, vec![0.0, 0.0, 1.0, 0.0]); // one-hot
//! let val = unflatten_discrete(&space, &flat);
//! assert_eq!(val, 2);
//! ```

use std::collections::HashMap;

use super::{BoundedSpace, DictSpace, Discrete, MultiDiscrete, Space};

/// Flatten a [`Discrete`] value to a one-hot encoded `Vec<f32>`.
///
/// The output has length `space.n`, with `1.0` at index `value - space.start`.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn flatten_discrete(space: &Discrete, value: i64) -> Vec<f32> {
    let mut out = vec![0.0_f32; space.n as usize];
    let idx = (value - space.start) as usize;
    if idx < out.len() {
        out[idx] = 1.0;
    }
    out
}

/// Unflatten a one-hot `Vec<f32>` back to a [`Discrete`] value.
///
/// Returns the index of the maximum element plus `space.start`.
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub fn unflatten_discrete(space: &Discrete, flat: &[f32]) -> i64 {
    let idx = flat
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);
    space.start + idx as i64
}

/// Flatten a [`MultiDiscrete`] value to concatenated one-hot vectors.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn flatten_multi_discrete(space: &MultiDiscrete, value: &[i64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(space.flatdim());
    for (&v, (&n, &s)) in value
        .iter()
        .zip(space.nvec().iter().zip(space.start().iter()))
    {
        let mut one_hot = vec![0.0_f32; n as usize];
        let idx = (v - s) as usize;
        if idx < one_hot.len() {
            one_hot[idx] = 1.0;
        }
        out.extend_from_slice(&one_hot);
    }
    out
}

/// Unflatten concatenated one-hot vectors back to a [`MultiDiscrete`] value.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub fn unflatten_multi_discrete(space: &MultiDiscrete, flat: &[f32]) -> Vec<i64> {
    let mut result = Vec::with_capacity(space.nvec().len());
    let mut offset = 0;
    for (&n, &s) in space.nvec().iter().zip(space.start().iter()) {
        let n_usize = n as usize;
        let chunk = &flat[offset..offset + n_usize];
        let idx = chunk
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        result.push(s + idx as i64);
        offset += n_usize;
    }
    result
}

/// Flatten a [`MultiBinary`](super::MultiBinary) value to `Vec<f32>` (cast `u8` → `f32`).
#[must_use]
pub fn flatten_multi_binary(value: &[u8]) -> Vec<f32> {
    value.iter().map(|&b| f32::from(b)).collect()
}

/// Unflatten a flat vector back to a [`MultiBinary`](super::MultiBinary) value.
///
/// Values ≥ 0.5 become 1, otherwise 0.
#[must_use]
pub fn unflatten_multi_binary(flat: &[f32]) -> Vec<u8> {
    flat.iter().map(|&v| u8::from(v >= 0.5)).collect()
}

/// Flatten a [`BoundedSpace`] value (identity — already a `Vec<f32>`).
///
/// This is a no-op clone, provided for API completeness so that generic
/// flatten pipelines can treat all space types uniformly.
#[must_use]
pub fn flatten_bounded(value: &[f32]) -> Vec<f32> {
    value.to_vec()
}

/// Unflatten a flat vector back to a [`BoundedSpace`] value (identity).
#[must_use]
pub fn unflatten_bounded(flat: &[f32]) -> Vec<f32> {
    flat.to_vec()
}

/// Flatten a homogeneous [`DictSpace<BoundedSpace>`] value by concatenating
/// all entries in the space's insertion order.
///
/// Returns a single `Vec<f32>` containing all sub-values concatenated.
#[must_use]
pub fn flatten_dict_bounded<S: ::std::hash::BuildHasher>(
    space: &DictSpace<BoundedSpace>,
    value: &HashMap<String, Vec<f32>, S>,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(space.flatdim());
    for (key, _) in space.iter() {
        if let Some(v) = value.get(key) {
            out.extend_from_slice(v);
        }
    }
    out
}

/// Unflatten a flat vector back to a [`DictSpace<BoundedSpace>`] value.
///
/// Splits the flat vector according to each sub-space's `flatdim`,
/// following the space's insertion order.
#[must_use]
pub fn unflatten_dict_bounded(
    space: &DictSpace<BoundedSpace>,
    flat: &[f32],
) -> HashMap<String, Vec<f32>> {
    let mut result = HashMap::new();
    let mut offset = 0;
    for (key, sub) in space.iter() {
        let dim = sub.flatdim();
        let chunk = flat[offset..offset + dim].to_vec();
        result.insert(key.to_owned(), chunk);
        offset += dim;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::MultiDiscrete;

    #[test]
    #[allow(
        clippy::float_cmp,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    fn discrete_flatten_unflatten() {
        let space = Discrete::new(5);
        for v in 0..5_i64 {
            let flat = flatten_discrete(&space, v);
            assert_eq!(flat.len(), 5);
            assert_eq!(flat[v as usize], 1.0);
            assert_eq!(unflatten_discrete(&space, &flat), v);
        }
    }

    #[test]
    fn discrete_with_start() {
        let space = Discrete::with_start(3, -1); // {-1, 0, 1}
        let flat = flatten_discrete(&space, 0);
        assert_eq!(flat, vec![0.0, 1.0, 0.0]);
        assert_eq!(unflatten_discrete(&space, &flat), 0);
    }

    #[test]
    fn multi_discrete_flatten_unflatten() {
        let space = MultiDiscrete::new(vec![3, 2]).unwrap();
        let value = vec![2, 1];
        let flat = flatten_multi_discrete(&space, &value);
        assert_eq!(flat.len(), 5);
        assert_eq!(flat, vec![0.0, 0.0, 1.0, 0.0, 1.0]);
        let back = unflatten_multi_discrete(&space, &flat);
        assert_eq!(back, value);
    }

    #[test]
    fn multi_binary_flatten_unflatten() {
        let value = vec![1_u8, 0, 1, 1];
        let flat = flatten_multi_binary(&value);
        assert_eq!(flat, vec![1.0, 0.0, 1.0, 1.0]);
        let back = unflatten_multi_binary(&flat);
        assert_eq!(back, value);
    }

    #[test]
    fn bounded_flatten_unflatten() {
        let value = vec![1.0_f32, -0.5, 3.14];
        let flat = flatten_bounded(&value);
        assert_eq!(flat, value);
        let back = unflatten_bounded(&flat);
        assert_eq!(back, value);
    }

    #[test]
    fn dict_bounded_flatten_unflatten() {
        use crate::space::{BoundedSpace, DictSpace};

        let space = DictSpace::new(vec![
            (
                "pos".into(),
                BoundedSpace::new(vec![-1.0, -1.0], vec![1.0, 1.0]).unwrap(),
            ),
            (
                "vel".into(),
                BoundedSpace::new(vec![0.0, 0.0, 0.0], vec![10.0, 10.0, 10.0]).unwrap(),
            ),
        ]);

        let mut value = HashMap::new();
        value.insert("pos".to_owned(), vec![0.5, -0.3]);
        value.insert("vel".to_owned(), vec![1.0, 2.0, 3.0]);

        let flat = flatten_dict_bounded(&space, &value);
        assert_eq!(flat.len(), 5); // 2 + 3

        let back = unflatten_dict_bounded(&space, &flat);
        assert_eq!(back["pos"], vec![0.5, -0.3]);
        assert_eq!(back["vel"], vec![1.0, 2.0, 3.0]);
    }
}
