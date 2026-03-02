//! Global environment registry and factory functions.
//!
//! Provides [`make`], [`register`], and [`spec`] — the Rust equivalent of
//! `gymnasium.make()`, `gymnasium.register()`, and `gymnasium.spec()`.
//!
//! Environments are identified by string ids such as `"CartPole-v1"` and
//! created through type-erased factory closures stored in a thread-safe
//! global registry.

mod dyn_env;
mod dyn_value;

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

pub use dyn_env::DynEnv;
pub use dyn_value::DynValue;

use crate::env::RenderMode;
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

/// Create a synchronous vectorized environment from a registered id.
///
/// Spawns `num_envs` copies of the environment identified by `id`, each with
/// `RenderMode::None`, and wraps them in a [`DynVectorEnv`].
///
/// # Errors
///
/// Returns [`Error::NotRegistered`] if the id is not found, or any
/// sub-environment factory error.
///
/// # Panics
///
/// Panics if the internal registry lock is poisoned.
pub fn make_vec(id: &str, num_envs: usize) -> Result<DynVectorEnv> {
    make_vec_with(id, num_envs, RenderMode::None)
}

/// Create a synchronous vectorized environment with a specific render mode.
///
/// # Errors
///
/// Returns [`Error::NotRegistered`] if the id is not found, or any
/// sub-environment factory error.
///
/// # Panics
///
/// Panics if the internal registry lock is poisoned.
pub fn make_vec_with(id: &str, num_envs: usize, render_mode: RenderMode) -> Result<DynVectorEnv> {
    if num_envs == 0 {
        return Err(Error::InvalidSpace {
            reason: "make_vec requires at least 1 environment".to_owned(),
        });
    }
    let envs: Result<Vec<_>> = (0..num_envs).map(|_| make_with(id, render_mode)).collect();
    Ok(DynVectorEnv::new(envs?))
}

/// A synchronous vectorized environment backed by type-erased [`DynEnv`] instances.
///
/// Created via [`make_vec`] or [`make_vec_with`]. Mirrors [`SyncVectorEnv`](crate::vector::SyncVectorEnv)
/// but operates on [`DynValue`] observations and actions.
#[derive(Debug)]
pub struct DynVectorEnv {
    envs: Vec<Box<dyn DynEnv>>,
    needs_reset: Vec<bool>,
}

impl DynVectorEnv {
    fn new(envs: Vec<Box<dyn DynEnv>>) -> Self {
        let n = envs.len();
        Self {
            envs,
            needs_reset: vec![false; n],
        }
    }

    /// The number of sub-environments.
    #[must_use]
    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Reset all sub-environments.
    ///
    /// If `seed` is provided, each sub-env receives `seed + i` as its seed.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during reset.
    pub fn reset(&mut self, seed: Option<u64>) -> Result<crate::vector::VecResetResult<DynValue>> {
        let mut obs = Vec::with_capacity(self.envs.len());
        let mut infos = Vec::with_capacity(self.envs.len());

        for (i, env) in self.envs.iter_mut().enumerate() {
            let s = seed.map(|s| s + i as u64);
            let r = env.reset_dyn(s)?;
            obs.push(r.obs);
            infos.push(r.info);
        }

        self.needs_reset.fill(false);
        Ok(crate::vector::VecResetResult { obs, infos })
    }

    /// Step all sub-environments with one action per env.
    ///
    /// # Errors
    ///
    /// Returns an error if `actions.len() != num_envs()` or any sub-env fails.
    pub fn step(&mut self, actions: &[DynValue]) -> Result<crate::vector::VecStepResult<DynValue>> {
        if actions.len() != self.envs.len() {
            return Err(Error::InvalidAction {
                reason: format!(
                    "expected {} actions, got {}",
                    self.envs.len(),
                    actions.len()
                ),
            });
        }

        let n = self.envs.len();
        let mut obs = Vec::with_capacity(n);
        let mut rewards = Vec::with_capacity(n);
        let mut terminated = Vec::with_capacity(n);
        let mut truncated = Vec::with_capacity(n);
        let mut infos = Vec::with_capacity(n);

        for (i, (env, action)) in self.envs.iter_mut().zip(actions.iter()).enumerate() {
            if self.needs_reset[i] {
                env.reset_dyn(None)?;
            }

            let r = env.step_dyn(action)?;
            let done = r.terminated || r.truncated;
            self.needs_reset[i] = done;

            obs.push(r.obs);
            rewards.push(r.reward);
            terminated.push(r.terminated);
            truncated.push(r.truncated);
            infos.push(r.info);
        }

        Ok(crate::vector::VecStepResult {
            obs,
            rewards,
            terminated,
            truncated,
            infos,
        })
    }

    /// Render all sub-environments.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during rendering.
    pub fn render(&mut self) -> Result<Vec<crate::env::RenderFrame>> {
        self.envs.iter_mut().map(|e| e.render_dyn()).collect()
    }

    /// Close all sub-environments.
    pub fn close(&mut self) {
        for env in &mut self.envs {
            env.close_dyn();
        }
    }
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

    let _ = register(EnvSpec {
        id: "MountainCarContinuous-v0".to_owned(),
        max_episode_steps: Some(999),
        reward_threshold: Some(90.0),
        factory: Box::new(|rm| {
            use crate::envs::classic_control::{
                ContinuousMountainCarConfig, ContinuousMountainCarEnv,
            };
            let env = ContinuousMountainCarEnv::new(ContinuousMountainCarConfig {
                render_mode: rm,
                ..ContinuousMountainCarConfig::default()
            })?;
            Ok(Box::new(env))
        }),
    });

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

    let _ = register(EnvSpec {
        id: "FrozenLake-v1".to_owned(),
        max_episode_steps: Some(100),
        reward_threshold: Some(0.70),
        factory: Box::new(|rm| {
            use crate::envs::toy_text::{FrozenLakeConfig, FrozenLakeEnv};
            let env = FrozenLakeEnv::new(FrozenLakeConfig {
                render_mode: rm,
                ..FrozenLakeConfig::default()
            })?;
            Ok(Box::new(env))
        }),
    });

    let _ = register(EnvSpec {
        id: "FrozenLake8x8-v1".to_owned(),
        max_episode_steps: Some(200),
        reward_threshold: Some(0.85),
        factory: Box::new(|rm| {
            use crate::envs::toy_text::{FrozenLakeConfig, FrozenLakeEnv, MAP_8X8};
            let env = FrozenLakeEnv::new(FrozenLakeConfig {
                desc: MAP_8X8.iter().map(|s| (*s).to_owned()).collect(),
                render_mode: rm,
                ..FrozenLakeConfig::default()
            })?;
            Ok(Box::new(env))
        }),
    });

    let _ = register(EnvSpec {
        id: "Taxi-v3".to_owned(),
        max_episode_steps: Some(200),
        reward_threshold: Some(8.0),
        factory: Box::new(|rm| {
            use crate::envs::toy_text::{TaxiConfig, TaxiEnv};
            let env = TaxiEnv::new(TaxiConfig { render_mode: rm });
            Ok(Box::new(env))
        }),
    });

    let _ = register(EnvSpec {
        id: "CliffWalking-v1".to_owned(),
        max_episode_steps: None,
        reward_threshold: None,
        factory: Box::new(|rm| {
            use crate::envs::toy_text::{CliffWalkingConfig, CliffWalkingEnv};
            let env = CliffWalkingEnv::new(CliffWalkingConfig { render_mode: rm });
            Ok(Box::new(env))
        }),
    });

    let _ = register(EnvSpec {
        id: "Blackjack-v1".to_owned(),
        max_episode_steps: None,
        reward_threshold: None,
        factory: Box::new(|rm| {
            use crate::envs::toy_text::{BlackjackConfig, BlackjackEnv};
            let env = BlackjackEnv::new(BlackjackConfig {
                sab: true,
                natural: false,
                render_mode: rm,
            });
            Ok(Box::new(env))
        }),
    });
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

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
        assert!(ids.contains(&"FrozenLake-v1".to_owned()));
        assert!(ids.contains(&"FrozenLake8x8-v1".to_owned()));
        assert!(ids.contains(&"Taxi-v3".to_owned()));
        assert!(ids.contains(&"CliffWalking-v1".to_owned()));
        assert!(ids.contains(&"Blackjack-v1".to_owned()));
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

    #[test]
    fn make_frozen_lake() {
        setup();
        let mut env = make("FrozenLake-v1").unwrap();
        let r = env.reset_dyn(Some(0)).unwrap();
        assert!(matches!(r.obs, DynValue::Discrete(_)));
    }

    #[test]
    fn make_taxi() {
        setup();
        let mut env = make("Taxi-v3").unwrap();
        let r = env.reset_dyn(Some(0)).unwrap();
        assert!(matches!(r.obs, DynValue::Discrete(_)));
    }

    #[test]
    fn make_blackjack() {
        setup();
        let mut env = make("Blackjack-v1").unwrap();
        let r = env.reset_dyn(Some(42)).unwrap();
        if let DynValue::Tuple(elems) = &r.obs {
            assert_eq!(elems.len(), 3);
        } else {
            panic!("expected Tuple observation, got {:?}", r.obs);
        }
    }
}
