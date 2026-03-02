//! Structured environment identifier parsing.
//!
//! Mirrors [Gymnasium `ENV_ID_RE`](https://gymnasium.farama.org/api/registry/)
//! supporting the format `namespace/Name-vN` where namespace and version are
//! optional.
//!
//! # Examples
//!
//! ```
//! use gmgn::registry::EnvId;
//!
//! let id = EnvId::parse("CartPole-v1").unwrap();
//! assert_eq!(id.namespace(), None);
//! assert_eq!(id.name(), "CartPole");
//! assert_eq!(id.version(), Some(1));
//! assert_eq!(id.to_string(), "CartPole-v1");
//!
//! let id = EnvId::parse("custom/MyEnv-v0").unwrap();
//! assert_eq!(id.namespace(), Some("custom"));
//! assert_eq!(id.name(), "MyEnv");
//! assert_eq!(id.version(), Some(0));
//!
//! let id = EnvId::parse("SimpleEnv").unwrap();
//! assert_eq!(id.namespace(), None);
//! assert_eq!(id.name(), "SimpleEnv");
//! assert_eq!(id.version(), None);
//! ```

use crate::error::{Error, Result};

/// A parsed environment identifier with optional namespace and version.
///
/// Format: `[namespace/]name[-vN]`
///
/// - **namespace**: optional prefix separated by `/` (e.g. `"custom"`, `"my_pkg"`)
/// - **name**: required environment name (e.g. `"CartPole"`, `"FrozenLake8x8"`)
/// - **version**: optional integer version after `-v` (e.g. `1` in `"-v1"`)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnvId {
    /// Optional namespace prefix.
    namespace: Option<String>,
    /// Environment name (without namespace or version).
    name: String,
    /// Optional version number.
    version: Option<u32>,
}

impl EnvId {
    /// Parse a string into a structured [`EnvId`].
    ///
    /// Accepted formats:
    /// - `"Name"` → no namespace, no version
    /// - `"Name-vN"` → no namespace, version N
    /// - `"ns/Name"` → namespace "ns", no version
    /// - `"ns/Name-vN"` → namespace "ns", version N
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpace`] if the string is empty or contains
    /// an invalid version suffix.
    pub fn parse(id: &str) -> Result<Self> {
        if id.is_empty() {
            return Err(Error::InvalidSpace {
                reason: "environment id must not be empty".to_owned(),
            });
        }

        // Split namespace from the rest.
        let (namespace, rest) = if let Some(slash_pos) = id.find('/') {
            let ns = &id[..slash_pos];
            let rest = &id[slash_pos + 1..];
            if ns.is_empty() || rest.is_empty() {
                return Err(Error::InvalidSpace {
                    reason: format!("invalid environment id: {id:?}"),
                });
            }
            (Some(ns.to_owned()), rest)
        } else {
            (None, id)
        };

        // Split version from name: look for the last "-v" followed by digits.
        let (name, version) = rest.rfind("-v").map_or_else(
            || (rest.to_owned(), None),
            |v_pos| {
                let suffix = &rest[v_pos + 2..];
                suffix.parse::<u32>().map_or_else(
                    |_| (rest.to_owned(), None),
                    |ver| (rest[..v_pos].to_owned(), Some(ver)),
                )
            },
        );

        if name.is_empty() {
            return Err(Error::InvalidSpace {
                reason: format!("environment id has empty name: {id:?}"),
            });
        }

        Ok(Self {
            namespace,
            name,
            version,
        })
    }

    /// The optional namespace.
    #[must_use]
    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    /// The environment name (without namespace or version).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The optional version number.
    #[must_use]
    pub const fn version(&self) -> Option<u32> {
        self.version
    }

    /// Reconstruct the full id string.
    #[must_use]
    pub fn full_id(&self) -> String {
        self.to_string()
    }
}

impl std::fmt::Display for EnvId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref ns) = self.namespace {
            write!(f, "{ns}/")?;
        }
        write!(f, "{}", self.name)?;
        if let Some(v) = self.version {
            write!(f, "-v{v}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_name_only() {
        let id = EnvId::parse("SimpleEnv").unwrap();
        assert_eq!(id.namespace(), None);
        assert_eq!(id.name(), "SimpleEnv");
        assert_eq!(id.version(), None);
        assert_eq!(id.to_string(), "SimpleEnv");
    }

    #[test]
    fn parse_name_with_version() {
        let id = EnvId::parse("CartPole-v1").unwrap();
        assert_eq!(id.namespace(), None);
        assert_eq!(id.name(), "CartPole");
        assert_eq!(id.version(), Some(1));
        assert_eq!(id.to_string(), "CartPole-v1");
    }

    #[test]
    fn parse_namespace_name_version() {
        let id = EnvId::parse("custom/MyEnv-v0").unwrap();
        assert_eq!(id.namespace(), Some("custom"));
        assert_eq!(id.name(), "MyEnv");
        assert_eq!(id.version(), Some(0));
        assert_eq!(id.to_string(), "custom/MyEnv-v0");
    }

    #[test]
    fn parse_namespace_no_version() {
        let id = EnvId::parse("pkg/Env").unwrap();
        assert_eq!(id.namespace(), Some("pkg"));
        assert_eq!(id.name(), "Env");
        assert_eq!(id.version(), None);
    }

    #[test]
    fn parse_name_with_hyphen() {
        let id = EnvId::parse("FrozenLake8x8-v1").unwrap();
        assert_eq!(id.name(), "FrozenLake8x8");
        assert_eq!(id.version(), Some(1));
    }

    #[test]
    fn parse_name_with_embedded_dash_v() {
        // "-v" not followed by digits stays as part of the name.
        let id = EnvId::parse("My-v-Env").unwrap();
        assert_eq!(id.name(), "My-v-Env");
        assert_eq!(id.version(), None);
    }

    #[test]
    fn parse_high_version() {
        let id = EnvId::parse("Env-v42").unwrap();
        assert_eq!(id.version(), Some(42));
    }

    #[test]
    fn empty_string_errors() {
        assert!(EnvId::parse("").is_err());
    }

    #[test]
    fn empty_namespace_errors() {
        assert!(EnvId::parse("/Name-v1").is_err());
    }

    #[test]
    fn empty_name_after_namespace_errors() {
        assert!(EnvId::parse("ns/").is_err());
    }

    #[test]
    fn roundtrip() {
        for id_str in [
            "CartPole-v1",
            "custom/MyEnv-v0",
            "SimpleEnv",
            "ns/NoVer",
            "MountainCarContinuous-v0",
        ] {
            let id = EnvId::parse(id_str).unwrap();
            assert_eq!(id.to_string(), id_str);
        }
    }
}
