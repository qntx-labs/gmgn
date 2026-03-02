//! Dynamically-typed observation/action values for the registry interface.

use std::collections::HashMap;

use crate::error::{Error, Result};

/// A dynamically-typed reinforcement learning environment value.
///
/// Used for type-erased observations and actions flowing through the
/// [`DynEnv`](super::DynEnv) interface.
#[derive(Debug, Clone, PartialEq)]
pub enum DynValue {
    /// Flat continuous vector (e.g. from [`BoundedSpace`](crate::space::BoundedSpace)).
    Continuous(Vec<f32>),
    /// Single discrete integer (e.g. from [`Discrete`](crate::space::Discrete)).
    Discrete(i64),
    /// Heterogeneous tuple of values (e.g. from [`Tuple2`](crate::space::Tuple2) /
    /// [`Tuple3`](crate::space::Tuple3)).
    Tuple(Vec<Self>),
    /// Multi-discrete integer vector (e.g. from [`MultiDiscrete`](crate::space::MultiDiscrete)).
    MultiDiscrete(Vec<i64>),
    /// Multi-binary byte vector (e.g. from [`MultiBinary`](crate::space::MultiBinary)).
    MultiBinary(Vec<u8>),
    /// Named dictionary of values (e.g. from `DictSpace`).
    Dict(HashMap<String, Self>),
    /// Text string value (e.g. from `TextSpace`).
    Text(String),
}

impl From<Vec<f32>> for DynValue {
    fn from(v: Vec<f32>) -> Self {
        Self::Continuous(v)
    }
}

impl From<i64> for DynValue {
    fn from(v: i64) -> Self {
        Self::Discrete(v)
    }
}

impl From<(i64, i64, i64)> for DynValue {
    fn from(v: (i64, i64, i64)) -> Self {
        Self::Tuple(vec![
            Self::Discrete(v.0),
            Self::Discrete(v.1),
            Self::Discrete(v.2),
        ])
    }
}

impl From<Vec<i64>> for DynValue {
    fn from(v: Vec<i64>) -> Self {
        Self::MultiDiscrete(v)
    }
}

impl From<Vec<u8>> for DynValue {
    fn from(v: Vec<u8>) -> Self {
        Self::MultiBinary(v)
    }
}

impl From<String> for DynValue {
    fn from(v: String) -> Self {
        Self::Text(v)
    }
}

impl From<HashMap<String, Self>> for DynValue {
    fn from(v: HashMap<String, Self>) -> Self {
        Self::Dict(v)
    }
}

impl TryFrom<DynValue> for Vec<f32> {
    type Error = Error;
    fn try_from(v: DynValue) -> Result<Self> {
        match v {
            DynValue::Continuous(c) => Ok(c),
            other => Err(Error::TypeMismatch {
                reason: format!("expected Continuous, got {other:?}"),
            }),
        }
    }
}

impl TryFrom<DynValue> for i64 {
    type Error = Error;
    fn try_from(v: DynValue) -> Result<Self> {
        match v {
            DynValue::Discrete(d) => Ok(d),
            other => Err(Error::TypeMismatch {
                reason: format!("expected Discrete, got {other:?}"),
            }),
        }
    }
}

impl TryFrom<DynValue> for (i64, i64, i64) {
    type Error = Error;
    fn try_from(v: DynValue) -> Result<Self> {
        match v {
            DynValue::Tuple(elems) if elems.len() == 3 => {
                let a = i64::try_from(elems[0].clone())?;
                let b = i64::try_from(elems[1].clone())?;
                let c = i64::try_from(elems[2].clone())?;
                Ok((a, b, c))
            }
            other => Err(Error::TypeMismatch {
                reason: format!("expected Tuple(3×Discrete), got {other:?}"),
            }),
        }
    }
}

impl TryFrom<DynValue> for Vec<i64> {
    type Error = Error;
    fn try_from(v: DynValue) -> Result<Self> {
        match v {
            DynValue::MultiDiscrete(d) => Ok(d),
            other => Err(Error::TypeMismatch {
                reason: format!("expected MultiDiscrete, got {other:?}"),
            }),
        }
    }
}

impl TryFrom<DynValue> for Vec<u8> {
    type Error = Error;
    fn try_from(v: DynValue) -> Result<Self> {
        match v {
            DynValue::MultiBinary(b) => Ok(b),
            other => Err(Error::TypeMismatch {
                reason: format!("expected MultiBinary, got {other:?}"),
            }),
        }
    }
}

impl TryFrom<DynValue> for String {
    type Error = Error;
    fn try_from(v: DynValue) -> Result<Self> {
        match v {
            DynValue::Text(s) => Ok(s),
            other => Err(Error::TypeMismatch {
                reason: format!("expected Text, got {other:?}"),
            }),
        }
    }
}

impl<S: ::std::hash::BuildHasher + Default> TryFrom<DynValue> for HashMap<String, DynValue, S> {
    type Error = Error;
    fn try_from(v: DynValue) -> Result<Self> {
        match v {
            DynValue::Dict(d) => Ok(d.into_iter().collect()),
            other => Err(Error::TypeMismatch {
                reason: format!("expected Dict, got {other:?}"),
            }),
        }
    }
}
