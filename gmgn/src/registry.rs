//! Global environment registry and factory functions.
//!
//! Provides [`make`], [`register`], and [`spec`] — the Rust equivalent of
//! `gymnasium.make()`, `gymnasium.register()`, and `gymnasium.spec()`.
//!
//! Environments are identified by string ids such as `"CartPole-v1"` and
//! created through type-erased factory closures stored in a thread-safe
//! global registry.

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use crate::env::RenderMode;
use crate::env::{RenderFrame, ResetResult, StepResult};
use crate::error::{Error, Result};

/// Metadata and factory for a registered environment.
///
/// Mirrors [Gymnasium `EnvSpec`](https://gymnasium.farama.org/api/registry/#gymnasium.envs.registration.EnvSpec).
pub struct EnvSpec {
    /// Unique identifier, e.g. `"CartPole-v1"`.
    pub id: String,
    /// Maximum steps per episode before truncation (`None` = unlimited).
    pub max_episode_steps: Option<u64>,
    /// Reward threshold that defines "solved" (`None` = unspecified).
    pub reward_threshold: Option<f64>,
    /// Factory that produces a boxed type-erased environment.
    factory: Box<dyn Fn(RenderMode) -> Result<Box<dyn DynEnv>> + Send + Sync>,
}

impl std::fmt::Debug for EnvSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvSpec")
            .field("id", &self.id)
            .field("max_episode_steps", &self.max_episode_steps)
            .field("reward_threshold", &self.reward_threshold)
            .finish_non_exhaustive()
    }
}

/// A dynamically-typed reinforcement learning environment value.
///
/// Used by [`ActionValue`] and [`ObsValue`] for type-erased observations and
/// actions flowing through the registry's [`DynEnv`] interface.
#[derive(Debug, Clone, PartialEq)]
pub enum DynValue {
    /// Flat continuous vector (e.g. from [`BoundedSpace`](crate::space::BoundedSpace)).
    Continuous(Vec<f32>),
    /// Single discrete integer (e.g. from [`Discrete`](crate::space::Discrete)).
    Discrete(i64),
}

/// A type-erased environment that operates on [`DynValue`].
///
/// Every concrete `Env` whose observation/action types can convert to/from
/// [`DynValue`] automatically implements `DynEnv` via the blanket impl below.
pub trait DynEnv: std::fmt::Debug {
    /// Advance one timestep.
    ///
    /// # Errors
    ///
    /// Returns an error if the action type mismatches or the inner env fails.
    fn step_dyn(&mut self, action: &DynValue) -> Result<StepResult<DynValue>>;

    /// Reset to initial state.
    ///
    /// # Errors
    ///
    /// Returns an error if the inner environment fails to reset.
    fn reset_dyn(&mut self, seed: Option<u64>) -> Result<ResetResult<DynValue>>;

    /// Render a frame.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering is unsupported or the env is not reset.
    fn render_dyn(&mut self) -> Result<RenderFrame>;

    /// Clean up resources.
    fn close_dyn(&mut self);
}

use crate::env::Env;

impl<E> DynEnv for E
where
    E: Env + std::fmt::Debug,
    E::Obs: Into<DynValue>,
    E::Act: TryFrom<DynValue, Error = Error>,
{
    fn step_dyn(&mut self, action: &DynValue) -> Result<StepResult<DynValue>> {
        let act = E::Act::try_from(action.clone())?;
        let r = self.step(&act)?;
        Ok(StepResult {
            obs: r.obs.into(),
            reward: r.reward,
            terminated: r.terminated,
            truncated: r.truncated,
            info: r.info,
        })
    }

    fn reset_dyn(&mut self, seed: Option<u64>) -> Result<ResetResult<DynValue>> {
        let r = self.reset(seed)?;
        Ok(ResetResult {
            obs: r.obs.into(),
            info: r.info,
        })
    }

    fn render_dyn(&mut self) -> Result<RenderFrame> {
        self.render()
    }

    fn close_dyn(&mut self) {
        self.close();
    }
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

impl TryFrom<DynValue> for Vec<f32> {
    type Error = Error;
    fn try_from(v: DynValue) -> Result<Self> {
        match v {
            DynValue::Continuous(c) => Ok(c),
            other @ DynValue::Discrete(_) => Err(Error::TypeMismatch {
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
            other @ DynValue::Continuous(_) => Err(Error::TypeMismatch {
                reason: format!("expected Discrete, got {other:?}"),
            }),
        }
    }
}

static REGISTRY: LazyLock<RwLock<HashMap<String, EnvSpec>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Register an environment specification.
///
/// # Errors
///
/// Returns [`Error::AlreadyRegistered`] if `spec.id` is already present.
///
/// # Panics
///
/// Panics if the internal registry lock is poisoned.
#[allow(clippy::unwrap_in_result)] // Lock poisoning is unrecoverable.
pub fn register(spec: EnvSpec) -> Result<()> {
    let mut reg = REGISTRY.write().expect("registry lock poisoned");
    if reg.contains_key(&spec.id) {
        return Err(Error::AlreadyRegistered { id: spec.id });
    }
    reg.insert(spec.id.clone(), spec);
    Ok(())
}

/// Create a new environment instance by its registered id.
///
/// # Errors
///
/// Returns [`Error::NotRegistered`] if the id is not found.
///
/// # Panics
///
/// Panics if the internal registry lock is poisoned.
pub fn make(id: &str) -> Result<Box<dyn DynEnv>> {
    make_with(id, RenderMode::None)
}

/// Create a new environment instance with a specific render mode.
///
/// # Errors
///
/// Returns [`Error::NotRegistered`] if the id is not found.
///
/// # Panics
///
/// Panics if the internal registry lock is poisoned.
#[allow(clippy::unwrap_in_result)] // Lock poisoning is unrecoverable.
pub fn make_with(id: &str, render_mode: RenderMode) -> Result<Box<dyn DynEnv>> {
    let reg = REGISTRY.read().expect("registry lock poisoned");
    let spec = reg
        .get(id)
        .ok_or_else(|| Error::NotRegistered { id: id.to_owned() })?;
    (spec.factory)(render_mode)
}

/// Look up the [`EnvSpec`] for a registered id.
///
/// Returns `None` if the id is not found.
///
/// # Panics
///
/// Panics if the internal registry lock is poisoned.
#[must_use]
pub fn spec(id: &str) -> Option<String> {
    let reg = REGISTRY.read().expect("registry lock poisoned");
    reg.get(id).map(|s| format!("{s:?}"))
}

/// List all registered environment ids.
///
/// # Panics
///
/// Panics if the internal registry lock is poisoned.
#[must_use]
pub fn list_registered() -> Vec<String> {
    let reg = REGISTRY.read().expect("registry lock poisoned");
    let mut ids: Vec<String> = reg.keys().cloned().collect();
    ids.sort();
    ids
}

/// Register all built-in environments.
///
/// Called automatically from `lib.rs` crate initialization.
pub fn register_builtins() {
    // CartPole-v1
    let _ = register(EnvSpec {
        id: "CartPole-v1".to_owned(),
        max_episode_steps: Some(500),
        reward_threshold: Some(475.0),
        factory: Box::new(|rm| {
            use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};
            let env = CartPoleEnv::new(CartPoleConfig {
                render_mode: rm,
                ..CartPoleConfig::default()
            })?;
            Ok(Box::new(env))
        }),
    });

    // MountainCar-v0
    let _ = register(EnvSpec {
        id: "MountainCar-v0".to_owned(),
        max_episode_steps: Some(200),
        reward_threshold: Some(-110.0),
        factory: Box::new(|rm| {
            use crate::envs::classic_control::{MountainCarConfig, MountainCarEnv};
            let env = MountainCarEnv::new(MountainCarConfig {
                render_mode: rm,
                ..MountainCarConfig::default()
            })?;
            Ok(Box::new(env))
        }),
    });

    // Pendulum-v1
    let _ = register(EnvSpec {
        id: "Pendulum-v1".to_owned(),
        max_episode_steps: Some(200),
        reward_threshold: None,
        factory: Box::new(|rm| {
            use crate::envs::classic_control::{PendulumConfig, PendulumEnv};
            let env = PendulumEnv::new(PendulumConfig {
                render_mode: rm,
                ..PendulumConfig::default()
            })?;
            Ok(Box::new(env))
        }),
    });

    // Acrobot-v1
    let _ = register(EnvSpec {
        id: "Acrobot-v1".to_owned(),
        max_episode_steps: Some(500),
        reward_threshold: Some(-100.0),
        factory: Box::new(|rm| {
            use crate::envs::classic_control::{AcrobotConfig, AcrobotEnv};
            let env = AcrobotEnv::new(AcrobotConfig { render_mode: rm })?;
            Ok(Box::new(env))
        }),
    });
}

#[cfg(test)]
#[allow(clippy::panic)] // Panics are acceptable in test assertions.
mod tests {
    use super::*;

    // Ensure builtins are registered before each test module run.
    fn setup() {
        register_builtins();
    }

    #[test]
    fn make_cartpole() {
        setup();
        let mut env = make("CartPole-v1").unwrap();
        let r = env.reset_dyn(Some(42)).unwrap();
        if let DynValue::Continuous(obs) = &r.obs {
            assert_eq!(obs.len(), 4);
        } else {
            panic!("expected Continuous observation");
        }
    }

    #[test]
    fn make_mountain_car() {
        setup();
        let mut env = make("MountainCar-v0").unwrap();
        let r = env.reset_dyn(Some(0)).unwrap();
        if let DynValue::Continuous(obs) = &r.obs {
            assert_eq!(obs.len(), 2);
        } else {
            panic!("expected Continuous observation");
        }
    }

    #[test]
    fn make_pendulum() {
        setup();
        let mut env = make("Pendulum-v1").unwrap();
        let r = env.reset_dyn(Some(0)).unwrap();
        if let DynValue::Continuous(obs) = &r.obs {
            assert_eq!(obs.len(), 3);
        } else {
            panic!("expected Continuous observation");
        }
    }

    #[test]
    fn make_unknown_errors() {
        setup();
        assert!(make("NonExistent-v99").is_err());
    }

    #[test]
    fn list_includes_builtins() {
        setup();
        let ids = list_registered();
        assert!(ids.contains(&"CartPole-v1".to_owned()));
        assert!(ids.contains(&"MountainCar-v0".to_owned()));
        assert!(ids.contains(&"Pendulum-v1".to_owned()));
    }

    #[test]
    fn step_through_dyn_env() {
        setup();
        let mut env = make("CartPole-v1").unwrap();
        env.reset_dyn(Some(42)).unwrap();
        let r = env.step_dyn(&DynValue::Discrete(1)).unwrap();
        if let DynValue::Continuous(obs) = &r.obs {
            assert_eq!(obs.len(), 4);
        } else {
            panic!("expected Continuous observation");
        }
    }
}
