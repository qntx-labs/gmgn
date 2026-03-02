//! Shared utilities for toy text environments.

use rand::RngExt as _;

use crate::rng::Rng;

/// Sample an index from a categorical (discrete) probability distribution.
///
/// Returns the sampled index in `0..probs.len()`.
#[allow(clippy::cast_possible_wrap)]
pub fn categorical_sample(probs: &[f64], rng: &mut Rng) -> usize {
    let r: f64 = rng.random_range(0.0..1.0);
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    probs.len() - 1
}
