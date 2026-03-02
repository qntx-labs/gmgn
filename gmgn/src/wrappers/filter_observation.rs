//! Filters a dictionary observation to a subset of keys.
//!
//! Mirrors [Gymnasium `FilterObservation`](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.FilterObservation).
//!
//! Useful for environments with [`DictSpace`](crate::space::DictSpace) observations
//! where only a subset of keys is relevant to the learning algorithm.

use std::collections::HashMap;

use crate::env::{Env, ResetResult, StepResult};
use crate::error::Result;
use crate::macros::delegate_env;
use crate::space::{DictSpace, Space};

/// Filters a [`DictSpace`] observation to retain only specified keys.
///
/// Both the observation and the observation space are filtered to contain
/// only the keys listed at construction time.
///
/// # Examples
///
/// ```rust,no_run
/// use std::collections::HashMap;
/// use gmgn::prelude::*;
/// use gmgn::wrappers::FilterObservation;
///
/// // Assume `env` has DictSpace<BoundedSpace> observation with keys "pos", "vel", "acc".
/// // FilterObservation::new(env, &["pos", "vel"]) keeps only "pos" and "vel".
/// ```
#[derive(Debug)]
pub struct FilterObservation<E, S>
where
    E: Env<Obs = HashMap<String, S::Element>, ObsSpace = DictSpace<S>>,
    S: Space,
{
    env: E,
    keys: Vec<String>,
    filtered_space: DictSpace<S>,
}

impl<E, S> FilterObservation<E, S>
where
    E: Env<Obs = HashMap<String, S::Element>, ObsSpace = DictSpace<S>>,
    S: Space + Clone,
{
    /// Wrap `env` to only keep the specified observation `keys`.
    ///
    /// Keys not present in the observation space are silently ignored.
    #[must_use]
    pub fn new(env: E, keys: &[&str]) -> Self {
        let key_set: Vec<String> = keys.iter().map(|&k| k.to_owned()).collect();
        let filtered_entries: Vec<(String, S)> = env
            .observation_space()
            .iter()
            .filter(|(k, _)| key_set.iter().any(|key| key == k))
            .map(|(k, s)| (k.to_owned(), s.clone()))
            .collect();
        let filtered_space = DictSpace::new(filtered_entries);
        Self {
            env,
            keys: key_set,
            filtered_space,
        }
    }

    /// The keys being retained.
    #[must_use]
    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    /// Borrow the inner environment.
    #[must_use]
    pub const fn inner(&self) -> &E {
        &self.env
    }

    /// Mutably borrow the inner environment.
    #[must_use]
    pub const fn inner_mut(&mut self) -> &mut E {
        &mut self.env
    }

    /// Unwrap and return the inner environment.
    #[must_use]
    pub fn into_inner(self) -> E {
        self.env
    }

    /// Filter a `HashMap` to retain only the configured keys.
    fn filter_obs(&self, mut obs: HashMap<String, S::Element>) -> HashMap<String, S::Element> {
        obs.retain(|k, _| self.keys.contains(k));
        obs
    }
}

impl<E, S> Env for FilterObservation<E, S>
where
    E: Env<Obs = HashMap<String, S::Element>, ObsSpace = DictSpace<S>>,
    S: Space + Clone,
{
    type Obs = HashMap<String, S::Element>;
    type Act = E::Act;
    type ObsSpace = DictSpace<S>;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let r = self.env.step(action)?;
        Ok(StepResult {
            obs: self.filter_obs(r.obs),
            reward: r.reward,
            terminated: r.terminated,
            truncated: r.truncated,
            info: r.info,
        })
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
        let r = self.env.reset(seed)?;
        Ok(ResetResult {
            obs: self.filter_obs(r.obs),
            info: r.info,
        })
    }

    fn observation_space(&self) -> &Self::ObsSpace {
        &self.filtered_space
    }

    delegate_env!(env, render, close, render_mode, action_space);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::{BoundedSpace, DictSpace, Space};

    /// A minimal dict-observation environment for testing.
    #[derive(Debug)]
    struct DictEnv {
        obs_space: DictSpace<BoundedSpace>,
        act_space: crate::space::Discrete,
        needs_reset: bool,
    }

    impl DictEnv {
        fn new() -> Self {
            let obs_space = DictSpace::new(vec![
                (
                    "pos".into(),
                    BoundedSpace::new(vec![-1.0], vec![1.0]).unwrap(),
                ),
                (
                    "vel".into(),
                    BoundedSpace::new(vec![-5.0], vec![5.0]).unwrap(),
                ),
                (
                    "acc".into(),
                    BoundedSpace::new(vec![-10.0], vec![10.0]).unwrap(),
                ),
            ]);
            Self {
                obs_space,
                act_space: crate::space::Discrete::new(2),
                needs_reset: true,
            }
        }
    }

    impl Env for DictEnv {
        type Obs = HashMap<String, Vec<f32>>;
        type Act = i64;
        type ObsSpace = DictSpace<BoundedSpace>;
        type ActSpace = crate::space::Discrete;

        fn step(&mut self, _action: &i64) -> Result<StepResult<Self::Obs>> {
            let mut rng = crate::rng::create_rng(Some(0));
            Ok(StepResult {
                obs: self.obs_space.sample(&mut rng),
                reward: 0.0,
                terminated: false,
                truncated: false,
                info: HashMap::default(),
            })
        }

        fn reset(&mut self, _seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
            self.needs_reset = false;
            let mut rng = crate::rng::create_rng(Some(0));
            Ok(ResetResult {
                obs: self.obs_space.sample(&mut rng),
                info: HashMap::default(),
            })
        }

        fn render(&mut self) -> Result<crate::env::RenderFrame> {
            Ok(crate::env::RenderFrame::None)
        }

        fn observation_space(&self) -> &DictSpace<BoundedSpace> {
            &self.obs_space
        }

        fn action_space(&self) -> &crate::space::Discrete {
            &self.act_space
        }

        fn render_mode(&self) -> &crate::env::RenderMode {
            &crate::env::RenderMode::None
        }
    }

    #[test]
    fn filters_to_subset() {
        let env = DictEnv::new();
        let mut env = FilterObservation::new(env, &["pos", "vel"]);
        let r = env.reset(Some(42)).unwrap();
        assert_eq!(r.obs.len(), 2);
        assert!(r.obs.contains_key("pos"));
        assert!(r.obs.contains_key("vel"));
        assert!(!r.obs.contains_key("acc"));
    }

    #[test]
    fn filtered_space_matches() {
        let env = DictEnv::new();
        let env = FilterObservation::new(env, &["pos"]);
        assert_eq!(env.observation_space().len(), 1);
        assert!(env.observation_space().get("pos").is_some());
    }

    #[test]
    fn step_also_filtered() {
        let env = DictEnv::new();
        let mut env = FilterObservation::new(env, &["acc"]);
        env.reset(Some(0)).unwrap();
        let r = env.step(&0).unwrap();
        assert_eq!(r.obs.len(), 1);
        assert!(r.obs.contains_key("acc"));
    }

    #[test]
    fn ignores_unknown_keys() {
        let env = DictEnv::new();
        let env = FilterObservation::new(env, &["pos", "nonexistent"]);
        // Only "pos" exists in the space, "nonexistent" is silently ignored.
        assert_eq!(env.observation_space().len(), 1);
    }
}
