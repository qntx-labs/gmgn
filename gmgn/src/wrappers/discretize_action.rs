//! Discretizes a continuous `BoundedSpace` action space into `Discrete`.
//!
//! Mirrors [Gymnasium `DiscretizeAction`](https://gymnasium.farama.org/api/wrappers/action_wrappers/#gymnasium.wrappers.DiscretizeAction).

use crate::env::{Env, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::{Error, Result};
use crate::space::{BoundedSpace, Discrete, Space};

/// Uniformly discretizes a continuous [`BoundedSpace`] action space into a
/// single [`Discrete`] space.
///
/// Given `bins` bins per dimension, the total number of discrete actions is
/// `bins ^ n_dims`. Each discrete action index is mapped to the center of
/// the corresponding bin in the original continuous space.
///
/// # Panics
///
/// The constructor panics if the action space contains non-finite bounds.
///
/// # Examples
///
/// ```rust,ignore
/// use gmgn::wrappers::DiscretizeAction;
///
/// // Pendulum has Box([-2.0], [2.0]) action space
/// let env = PendulumEnv::new(config)?;
/// let mut env = DiscretizeAction::new(env, 10)?; // 10 bins
/// // env.action_space() is now Discrete(10)
/// ```
#[derive(Debug)]
pub struct DiscretizeAction<E>
where
    E: Env<Act = Vec<f32>, ActSpace = BoundedSpace>,
{
    env: E,
    /// Number of bins per dimension.
    bins: usize,
    /// Number of dimensions in the original action space.
    n_dims: usize,
    /// Pre-computed bin centers for each dimension: `[n_dims][bins]`.
    bin_centers: Vec<Vec<f32>>,
    /// The discrete action space exposed to the agent.
    action_space: Discrete,
}

impl<E> DiscretizeAction<E>
where
    E: Env<Act = Vec<f32>, ActSpace = BoundedSpace>,
{
    /// Wrap `env` by discretizing its continuous action space.
    ///
    /// # Arguments
    ///
    /// * `bins` — Number of uniform bins per dimension. Must be ≥ 1.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpace`] if `bins` is zero or the action space
    /// has non-finite bounds.
    pub fn new(env: E, bins: usize) -> Result<Self> {
        if bins == 0 {
            return Err(Error::InvalidSpace {
                reason: "DiscretizeAction requires bins >= 1".to_owned(),
            });
        }

        let space = env.action_space();
        let n_dims = space.low.len();

        // Validate finite bounds.
        for (i, (&lo, &hi)) in space.low.iter().zip(space.high.iter()).enumerate() {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(Error::InvalidSpace {
                    reason: format!(
                        "DiscretizeAction requires finite bounds, dim {i}: [{lo}, {hi}]"
                    ),
                });
            }
        }

        // Pre-compute bin centers for each dimension.
        let bin_centers: Vec<Vec<f32>> = space
            .low
            .iter()
            .zip(space.high.iter())
            .map(|(&lo, &hi)| {
                let step = (hi - lo) / bins as f32;
                (0..bins)
                    .map(|b| step.mul_add(b as f32 + 0.5, lo))
                    .collect()
            })
            .collect();

        // Total discrete actions = bins ^ n_dims.
        #[allow(clippy::cast_possible_truncation)]
        let total = bins
            .checked_pow(n_dims as u32)
            .ok_or_else(|| Error::InvalidSpace {
                reason: format!("bins^n_dims overflow: {bins}^{n_dims}"),
            })?;

        let action_space = Discrete::new(total as u64);

        Ok(Self {
            env,
            bins,
            n_dims,
            bin_centers,
            action_space,
        })
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

    /// Convert a flat discrete index to a continuous action vector.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn decode_action(&self, flat_index: i64) -> Vec<f32> {
        let mut idx = flat_index as usize;
        let mut indices = vec![0_usize; self.n_dims];

        // Unflatten: least-significant dimension last (row-major).
        for d in (0..self.n_dims).rev() {
            indices[d] = idx % self.bins;
            idx /= self.bins;
        }

        indices
            .iter()
            .enumerate()
            .map(|(d, &b)| self.bin_centers[d][b.min(self.bins - 1)])
            .collect()
    }
}

impl<E> Env for DiscretizeAction<E>
where
    E: Env<Act = Vec<f32>, ActSpace = BoundedSpace>,
{
    type Obs = E::Obs;
    type Act = i64;
    type ObsSpace = E::ObsSpace;
    type ActSpace = Discrete;

    fn step(&mut self, action: &i64) -> Result<StepResult<Self::Obs>> {
        if !self.action_space.contains(action) {
            return Err(Error::InvalidAction {
                reason: format!(
                    "discrete action {action} not in {{0..{}}}",
                    self.action_space.n
                ),
            });
        }
        let continuous = self.decode_action(*action);
        self.env.step(&continuous)
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> {
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

    fn action_space(&self) -> &Discrete {
        &self.action_space
    }

    fn render_mode(&self) -> &RenderMode {
        self.env.render_mode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::classic_control::{PendulumConfig, PendulumEnv};
    use crate::rng::create_rng;

    #[test]
    fn discretize_pendulum() {
        let env = PendulumEnv::new(PendulumConfig::default()).unwrap();
        let mut env = DiscretizeAction::new(env, 10).unwrap();

        assert_eq!(env.action_space().n, 10);

        let r = env.reset(Some(42)).unwrap();
        assert!(env.observation_space().contains(&r.obs));

        // Step with a valid discrete action.
        let step = env.step(&3).unwrap();
        assert!(env.observation_space().contains(&step.obs));
    }

    #[test]
    fn invalid_action_errors() {
        let env = PendulumEnv::new(PendulumConfig::default()).unwrap();
        let mut env = DiscretizeAction::new(env, 5).unwrap();
        env.reset(Some(0)).unwrap();

        assert!(env.step(&5).is_err()); // 5 is out of {0..5}
        assert!(env.step(&-1).is_err());
    }

    #[test]
    fn zero_bins_errors() {
        let env = PendulumEnv::new(PendulumConfig::default()).unwrap();
        assert!(DiscretizeAction::new(env, 0).is_err());
    }

    #[test]
    fn bin_centers_correct() {
        let env = PendulumEnv::new(PendulumConfig::default()).unwrap();
        // Pendulum action space: Box([-2.0], [2.0])
        let env = DiscretizeAction::new(env, 4).unwrap();

        // 4 bins over [-2, 2]: centers at -1.5, -0.5, 0.5, 1.5
        let decoded = env.decode_action(0);
        assert!((decoded[0] - (-1.5)).abs() < 1e-5);
        let decoded = env.decode_action(3);
        assert!((decoded[0] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn multi_dim_discretization() {
        // ContinuousMountainCar has 1-dim action, so total = bins^1 = bins.
        use crate::envs::classic_control::{ContinuousMountainCarConfig, ContinuousMountainCarEnv};
        let env = ContinuousMountainCarEnv::new(ContinuousMountainCarConfig::default()).unwrap();
        let mut env = DiscretizeAction::new(env, 5).unwrap();

        assert_eq!(env.action_space().n, 5);
        env.reset(Some(0)).unwrap();
        let step = env.step(&2).unwrap();
        assert!(env.observation_space().contains(&step.obs));
    }

    #[test]
    fn sample_and_step() {
        let env = PendulumEnv::new(PendulumConfig::default()).unwrap();
        let mut env = DiscretizeAction::new(env, 7).unwrap();
        env.reset(Some(99)).unwrap();

        let mut rng = create_rng(Some(42));
        for _ in 0..20 {
            let action = env.action_space().sample(&mut rng);
            let r = env.step(&action).unwrap();
            assert!(env.observation_space().contains(&r.obs));
        }
    }
}
