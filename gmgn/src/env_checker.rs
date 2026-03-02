//! Environment validation utilities.
//!
//! Provides [`check_env`] to verify that an environment conforms to the
//! [`Env`](crate::env::Env) contract. Useful for catching common
//! implementation bugs in custom environments.
//!
//! Mirrors [Gymnasium `env_checker`](https://gymnasium.farama.org/api/utils/#gymnasium.utils.env_checker.check_env).

use crate::env::Env;
use crate::space::Space;

/// Severity level of a check result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckLevel {
    /// A non-critical issue that should be addressed.
    Warning,
    /// A critical issue that violates the `Env` contract.
    Error,
}

/// A single issue found by [`check_env`].
#[derive(Debug, Clone)]
pub struct CheckResult {
    /// Severity of the issue.
    pub level: CheckLevel,
    /// Human-readable description of the issue.
    pub message: String,
}

/// Validate that an environment conforms to the [`Env`] contract.
///
/// Performs the following checks:
/// - `reset()` succeeds and returns an observation within `observation_space`
/// - `step()` with a sampled action returns an observation within `observation_space`
/// - Rewards are finite
/// - `step()` before `reset()` returns an error
/// - (when `seed` is provided) Seeded resets are deterministic
///
/// Returns a list of issues found. An empty list means the environment passes
/// all checks.
///
/// # Examples
///
/// ```rust
/// use gmgn::env_checker::check_env;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
///
/// let mut env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let issues = check_env(&mut env, Some(42));
/// assert!(issues.is_empty(), "CartPole should pass all checks");
/// ```
pub fn check_env<E>(env: &mut E, seed: Option<u64>) -> Vec<CheckResult>
where
    E: Env,
    E::Obs: std::fmt::Debug + PartialEq,
    E::Act: std::fmt::Debug,
{
    let mut issues = Vec::new();

    // Check step before reset returns an error.
    check_step_before_reset(env, &mut issues);

    // Check reset produces valid observation.
    let reset_ok = check_reset(env, seed, &mut issues);
    if !reset_ok {
        return issues;
    }

    // Check step with sampled action.
    check_step(env, &mut issues);

    // Check seeded reset determinism.
    if let Some(s) = seed {
        check_seed_determinism(env, s, &mut issues);
    }

    issues
}

/// Verify that calling step before reset returns an error.
fn check_step_before_reset<E>(env: &mut E, issues: &mut Vec<CheckResult>)
where
    E: Env,
    E::Act: std::fmt::Debug,
{
    // Create a fresh environment to test step-before-reset.
    // We can't easily clone the env, so we test using a sampled action.
    // This check is best-effort: if the env doesn't error, we flag it.
    let mut rng = crate::rng::create_rng(Some(0));
    let action = env.action_space().sample(&mut rng);
    if env.step(&action).is_ok() {
        issues.push(CheckResult {
            level: CheckLevel::Warning,
            message: "step() succeeded before reset() — consider returning \
                      Error::ResetNeeded or wrapping with OrderEnforcing"
                .to_owned(),
        });
    }
}

/// Verify that reset produces an observation within the observation space.
fn check_reset<E>(env: &mut E, seed: Option<u64>, issues: &mut Vec<CheckResult>) -> bool
where
    E: Env,
    E::Obs: std::fmt::Debug,
{
    match env.reset(seed) {
        Ok(result) => {
            if !env.observation_space().contains(&result.obs) {
                issues.push(CheckResult {
                    level: CheckLevel::Error,
                    message: format!(
                        "reset() returned observation outside observation_space: {:?}",
                        result.obs
                    ),
                });
            }
            true
        }
        Err(e) => {
            issues.push(CheckResult {
                level: CheckLevel::Error,
                message: format!("reset() failed: {e}"),
            });
            false
        }
    }
}

/// Verify that step with a sampled action produces valid results.
fn check_step<E>(env: &mut E, issues: &mut Vec<CheckResult>)
where
    E: Env,
    E::Obs: std::fmt::Debug,
{
    let mut rng = crate::rng::create_rng(Some(1));
    let action = env.action_space().sample(&mut rng);

    match env.step(&action) {
        Ok(result) => {
            // Check observation is within space.
            if !env.observation_space().contains(&result.obs) {
                issues.push(CheckResult {
                    level: CheckLevel::Error,
                    message: format!(
                        "step() returned observation outside observation_space: {:?}",
                        result.obs
                    ),
                });
            }

            // Check reward is finite.
            if !result.reward.is_finite() {
                issues.push(CheckResult {
                    level: CheckLevel::Error,
                    message: format!("step() returned non-finite reward: {}", result.reward),
                });
            }
        }
        Err(e) => {
            issues.push(CheckResult {
                level: CheckLevel::Error,
                message: format!("step() with sampled action failed: {e}"),
            });
        }
    }
}

/// Verify that resetting with the same seed produces identical observations.
fn check_seed_determinism<E>(env: &mut E, seed: u64, issues: &mut Vec<CheckResult>)
where
    E: Env,
    E::Obs: std::fmt::Debug + PartialEq,
{
    let first = match env.reset(Some(seed)) {
        Ok(r) => r.obs,
        Err(e) => {
            issues.push(CheckResult {
                level: CheckLevel::Error,
                message: format!("seed determinism check: first reset failed: {e}"),
            });
            return;
        }
    };

    let second = match env.reset(Some(seed)) {
        Ok(r) => r.obs,
        Err(e) => {
            issues.push(CheckResult {
                level: CheckLevel::Error,
                message: format!("seed determinism check: second reset failed: {e}"),
            });
            return;
        }
    };

    if first != second {
        issues.push(CheckResult {
            level: CheckLevel::Error,
            message: format!(
                "reset(seed={seed}) is not deterministic: first={first:?}, second={second:?}"
            ),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};

    #[test]
    fn cartpole_passes_all_checks() {
        let mut env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let issues = check_env(&mut env, Some(42));
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.level == CheckLevel::Error)
            .collect();
        assert!(errors.is_empty(), "CartPole errors: {errors:?}");
    }

    #[test]
    fn frozen_lake_passes_checks() {
        use crate::envs::toy_text::{FrozenLakeConfig, FrozenLakeEnv};
        let mut env = FrozenLakeEnv::new(FrozenLakeConfig::default()).unwrap();
        let issues = check_env(&mut env, Some(42));
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.level == CheckLevel::Error)
            .collect();
        assert!(errors.is_empty(), "FrozenLake errors: {errors:?}");
    }

    #[test]
    fn pendulum_passes_checks() {
        use crate::envs::classic_control::{PendulumConfig, PendulumEnv};
        let mut env = PendulumEnv::new(PendulumConfig::default()).unwrap();
        let issues = check_env(&mut env, Some(42));
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.level == CheckLevel::Error)
            .collect();
        assert!(errors.is_empty(), "Pendulum errors: {errors:?}");
    }
}
