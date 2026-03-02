//! Random number generation for reproducible environments.

use rand::SeedableRng;

/// The default random number generator used throughout gmgn.
///
/// Uses PCG64 MCG (Permuted Congruential Generator), which provides
/// excellent statistical quality with minimal state (128 bits).
pub type Rng = rand_pcg::Pcg64Mcg;

/// Create a new RNG, optionally seeded for reproducibility.
///
/// If `seed` is `None`, the RNG is seeded from system entropy.
/// If `seed` is `Some(n)`, the RNG is deterministically seeded.
#[must_use]
pub fn create_rng(seed: Option<u64>) -> Rng {
    seed.map_or_else(|| Rng::from_rng(&mut rand::rng()), Rng::seed_from_u64)
}
