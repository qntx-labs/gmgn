//! Records cumulative episode reward and length into the `info` dict.
//!
//! Mirrors [Gymnasium `RecordEpisodeStatistics`](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordEpisodeStatistics).

use crate::env::{Env, InfoValue, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::Result;

/// Tracks per-episode reward and length, injecting them into the `info` dict
/// on the step that ends the episode (`terminated || truncated`).
///
/// Inserted keys:
/// - `"episode.r"` — cumulative reward ([`InfoValue::Float`]).
/// - `"episode.l"` — episode length ([`InfoValue::Int`]).
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::RecordEpisodeStatistics;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = RecordEpisodeStatistics::new(env);
/// ```
#[derive(Debug)]
pub struct RecordEpisodeStatistics<E: Env> {
    env: E,
    /// Cumulative reward for the current episode.
    episode_reward: f64,
    /// Steps taken in the current episode.
    episode_length: u64,
}

impl<E: Env> RecordEpisodeStatistics<E> {
    /// Wrap `env` to record episode statistics.
    #[must_use]
    pub const fn new(env: E) -> Self {
        Self {
            env,
            episode_reward: 0.0,
            episode_length: 0,
        }
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
}

impl<E: Env> Env for RecordEpisodeStatistics<E> {
    type Obs = E::Obs;
    type Act = E::Act;
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> {
        let mut result = self.env.step(action)?;

        self.episode_reward += result.reward;
        self.episode_length += 1;

        if result.terminated || result.truncated {
            result.info.insert(
                "episode.r".to_owned(),
                InfoValue::Float(self.episode_reward),
            );
            result.info.insert(
                "episode.l".to_owned(),
                InfoValue::Int(self.episode_length.cast_signed()),
            );
        }

        Ok(result)
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
        self.episode_reward = 0.0;
        self.episode_length = 0;
        self.env.reset(seed)
    }

    fn render(&mut self) -> Result<RenderFrame> {
        self.env.render()
    }

    fn close(&mut self) {
        self.env.close();
    }

    fn observation_space(&self) -> &Self::ObsSpace {
        self.env.observation_space()
    }

    fn action_space(&self) -> &Self::ActSpace {
        self.env.action_space()
    }

    fn render_mode(&self) -> &RenderMode {
        self.env.render_mode()
    }
}

#[cfg(test)]
#[allow(clippy::panic)] // Panics are acceptable in test assertions.
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};

    #[test]
    fn records_on_termination() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = RecordEpisodeStatistics::new(env);
        env.reset(Some(0)).unwrap();

        let mut last_info = None;
        // Push cart right until termination.
        for _ in 0..500 {
            let result = env.step(&1).unwrap();
            if result.terminated || result.truncated {
                last_info = Some(result.info);
                break;
            }
        }

        let info = last_info.expect("episode should terminate");
        assert!(info.contains_key("episode.r"));
        assert!(info.contains_key("episode.l"));

        if let InfoValue::Int(len) = info["episode.l"] {
            assert!(len > 0);
        } else {
            panic!("episode.l should be Int");
        }
    }

    #[test]
    fn no_stats_mid_episode() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = RecordEpisodeStatistics::new(env);
        env.reset(Some(42)).unwrap();

        let result = env.step(&0).unwrap();
        if !result.terminated && !result.truncated {
            assert!(!result.info.contains_key("episode.r"));
            assert!(!result.info.contains_key("episode.l"));
        }
    }

    #[test]
    fn reset_clears_accumulators() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = RecordEpisodeStatistics::new(env);

        env.reset(Some(0)).unwrap();
        env.step(&0).unwrap();
        env.reset(Some(1)).unwrap();

        // After reset, accumulators should be zero — step once and check.
        let r = env.step(&0).unwrap();
        if !r.terminated && !r.truncated {
            // Reward for a single CartPole step is 1.0 by default.
            assert!((env.episode_reward - 1.0).abs() < f64::EPSILON);
            assert_eq!(env.episode_length, 1);
        }
    }
}
