//! Text space — variable-length strings from a character set.
//!
//! Mirrors [Gymnasium `Text`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Text).
//!
//! # Examples
//!
//! ```
//! use gmgn::space::{TextSpace, Space};
//! use gmgn::rng::create_rng;
//!
//! let space = TextSpace::alphanumeric(1, 10);
//! let mut rng = create_rng(Some(42));
//! let sample = space.sample(&mut rng);
//! assert!(space.contains(&sample));
//! assert!((1..=10).contains(&sample.len()));
//! ```

use rand::RngExt;

use crate::rng::Rng;
use crate::space::{Space, SpaceInfo};

/// A space of variable-length strings drawn from a fixed character set.
///
/// Elements are UTF-8 [`String`]s whose length is in `[min_length, max_length]`
/// and whose characters are all members of `charset`.
#[derive(Debug, Clone)]
pub struct TextSpace {
    /// Minimum string length (inclusive).
    min_length: usize,
    /// Maximum string length (inclusive).
    max_length: usize,
    /// Allowed characters, stored as a sorted `Vec` for efficient sampling.
    charset: Vec<char>,
}

impl TextSpace {
    /// Create a text space with a custom character set.
    ///
    /// # Panics
    ///
    /// Panics if `charset` is empty or `min_length > max_length`.
    #[must_use]
    pub fn new(min_length: usize, max_length: usize, charset: &[char]) -> Self {
        assert!(
            !charset.is_empty(),
            "TextSpace charset must not be empty"
        );
        assert!(
            min_length <= max_length,
            "min_length ({min_length}) must be <= max_length ({max_length})"
        );
        let mut chars: Vec<char> = charset.to_vec();
        chars.sort_unstable();
        chars.dedup();
        Self {
            min_length,
            max_length,
            charset: chars,
        }
    }

    /// Create a text space using the default alphanumeric charset
    /// (`a-z`, `A-Z`, `0-9`).
    #[must_use]
    pub fn alphanumeric(min_length: usize, max_length: usize) -> Self {
        let charset: Vec<char> =
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                .chars()
                .collect();
        Self::new(min_length, max_length, &charset)
    }

    /// The minimum string length.
    #[must_use]
    pub const fn min_length(&self) -> usize {
        self.min_length
    }

    /// The maximum string length.
    #[must_use]
    pub const fn max_length(&self) -> usize {
        self.max_length
    }

    /// The allowed character set.
    #[must_use]
    pub fn charset(&self) -> &[char] {
        &self.charset
    }
}

impl Space for TextSpace {
    type Element = String;

    #[allow(clippy::cast_possible_truncation)]
    fn sample(&self, rng: &mut Rng) -> String {
        use rand::seq::IndexedRandom;
        let len = if self.min_length == self.max_length {
            self.min_length
        } else {
            rng.random_range(self.min_length..=self.max_length)
        };
        (0..len)
            .map(|_| *self.charset.choose(rng).expect("charset is non-empty"))
            .collect()
    }

    fn contains(&self, value: &String) -> bool {
        let len = value.chars().count();
        if len < self.min_length || len > self.max_length {
            return false;
        }
        value.chars().all(|c| self.charset.contains(&c))
    }

    fn shape(&self) -> &[usize] {
        &[]
    }

    fn flatdim(&self) -> usize {
        self.max_length
    }

    fn space_info(&self) -> SpaceInfo {
        // TextSpace doesn't have a dedicated SpaceInfo variant yet.
        // Represent as a bounded discrete range for now.
        SpaceInfo::Discrete {
            #[allow(clippy::cast_possible_wrap)]
            n: self.max_length as i64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::create_rng;

    #[test]
    fn alphanumeric_sample() {
        let space = TextSpace::alphanumeric(1, 10);
        let mut rng = create_rng(Some(42));
        for _ in 0..20 {
            let s = space.sample(&mut rng);
            assert!(space.contains(&s));
            assert!((1..=10).contains(&s.len()));
        }
    }

    #[test]
    fn fixed_length() {
        let space = TextSpace::alphanumeric(5, 5);
        let mut rng = create_rng(Some(0));
        let s = space.sample(&mut rng);
        assert_eq!(s.len(), 5);
        assert!(space.contains(&s));
    }

    #[test]
    fn custom_charset() {
        let space = TextSpace::new(3, 3, &['a', 'b', 'c']);
        let mut rng = create_rng(Some(42));
        let s = space.sample(&mut rng);
        assert_eq!(s.len(), 3);
        assert!(s.chars().all(|c| "abc".contains(c)));
        assert!(space.contains(&s));
    }

    #[test]
    fn rejects_invalid_chars() {
        let space = TextSpace::new(1, 5, &['a', 'b']);
        assert!(!space.contains(&"abc".to_string()));
    }

    #[test]
    fn rejects_wrong_length() {
        let space = TextSpace::alphanumeric(2, 4);
        assert!(!space.contains(&"a".to_string()));
        assert!(!space.contains(&"abcde".to_string()));
        assert!(space.contains(&"ab".to_string()));
    }

    #[test]
    #[should_panic(expected = "charset must not be empty")]
    fn empty_charset_panics() {
        let _ = TextSpace::new(1, 5, &[]);
    }

    #[test]
    #[should_panic(expected = "min_length")]
    fn inverted_length_panics() {
        let _ = TextSpace::alphanumeric(10, 5);
    }
}
