//! Running-mean observation normalization wrapper.
//!
//! Mirrors [Gymnasium `NormalizeObservation`](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.NormalizeObservation).

use crate::env::{Env, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::Result;

/// Normalizes observations using a running mean and variance estimate.
///
/// Each element of the observation vector is independently normalized to
/// approximately zero mean and unit variance via Welford's online algorithm.
///
/// Only applicable to environments whose `Obs` is `Vec<f32>`.
///
/// # Examples
///
/// ```rust,no_run
/// use gmgn::prelude::*;
/// use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
/// use gmgn::wrappers::NormalizeObservation;
///
/// let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
/// let mut env = NormalizeObservation::new(env, 1e-8);
/// ```
#[derive(Debug)]
pub struct NormalizeObservation<E: Env<Obs = Vec<f32>>> {
    env: E,
    /// Running count of observations seen.
    count: f64,
    /// Running per-element mean.
    mean: Vec<f64>,
    /// Running per-element variance (unnormalized sum of squared deviations).
    var: Vec<f64>,
    /// Small constant added to variance for numerical stability.
    epsilon: f64,
}

impl<E: Env<Obs = Vec<f32>>> NormalizeObservation<E> {
    /// Wrap `env` with running observation normalization.
    ///
    /// `epsilon` is a small constant (e.g. `1e-8`) added to the variance
    /// denominator for numerical stability.
    #[must_use]
    pub const fn new(env: E, epsilon: f64) -> Self {
        Self {
            env,
            count: 0.0,
            mean: Vec::new(),
            var: Vec::new(),
            epsilon,
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

    /// Update running statistics and return the normalized observation.
    #[allow(clippy::cast_possible_truncation)]
    fn normalize(&mut self, obs: &[f32]) -> Vec<f32> {
        // Lazy-initialize mean/var on first observation.
        if self.mean.is_empty() {
            self.mean = vec![0.0; obs.len()];
            self.var = vec![1.0; obs.len()];
        }

        // Welford's online algorithm.
        self.count += 1.0;
        for (i, &v) in obs.iter().enumerate() {
            let v = f64::from(v);
            let delta = v - self.mean[i];
            self.mean[i] += delta / self.count;
            let delta2 = v - self.mean[i];
            self.var[i] += delta * delta2;
        }

        // Normalize: (obs - mean) / sqrt(var / count + epsilon).
        obs.iter()
            .enumerate()
            .map(|(i, &v)| {
                let std = (self.var[i] / self.count + self.epsilon).sqrt();
                ((f64::from(v) - self.mean[i]) / std) as f32
            })
            .collect()
    }
}

impl<E: Env<Obs = Vec<f32>>> Env for NormalizeObservation<E> {
    type Obs = Vec<f32>;
    type Act = E::Act;
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn step(&mut self, action: &Self::Act) -> Result<StepResult<Vec<f32>>> {
        let mut result = self.env.step(action)?;
        result.obs = self.normalize(&result.obs);
        Ok(result)
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Vec<f32>>> {
        let mut result = self.env.reset(seed)?;
        result.obs = self.normalize(&result.obs);
        Ok(result)
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
mod tests {
    use super::*;
    use crate::envs::classic_control::{CartPoleConfig, CartPoleEnv};

    #[test]
    fn observations_are_normalized() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = NormalizeObservation::new(env, 1e-8);
        env.reset(Some(42)).unwrap();

        // After several steps, observations should be roughly normalized.
        for _ in 0..50 {
            let r = env.step(&0).unwrap();
            // Normalized observations should not blow up.
            for &v in &r.obs {
                assert!(v.is_finite(), "observation should be finite: {v}");
            }
            if r.terminated {
                env.reset(None).unwrap();
            }
        }
    }

    #[test]
    fn first_observation_is_finite() {
        let env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
        let mut env = NormalizeObservation::new(env, 1e-8);
        let r = env.reset(Some(0)).unwrap();
        for &v in &r.obs {
            assert!(v.is_finite());
        }
    }
}
