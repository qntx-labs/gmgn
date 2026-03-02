//! Inverted pendulum swing-up environment (continuous action).
//!
//! The goal is to apply torque to swing a pendulum into the upright position.
//!
//! Mirrors [Gymnasium `Pendulum-v1`](https://gymnasium.farama.org/environments/classic_control/pendulum/).

use std::collections::HashMap;
use std::f64::consts::PI;

use rand::RngExt as _;

use crate::env::{Env, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::{Error, Result};
#[cfg(feature = "render")]
use crate::render::{Canvas, RenderWindow};
use crate::rng::{self, Rng};
use crate::space::BoundedSpace;

const MAX_SPEED: f64 = 8.0;
const MAX_TORQUE: f64 = 2.0;
const DT: f64 = 0.05;
const MASS: f64 = 1.0;
const LENGTH: f64 = 1.0;

#[cfg(feature = "render")]
const SCREEN_DIM: u32 = 500;
#[cfg(feature = "render")]
const RENDER_FPS: usize = 30;

/// Configuration for [`PendulumEnv`].
#[derive(Debug, Clone, Copy)]
pub struct PendulumConfig {
    /// Acceleration due to gravity (m/s²).
    pub g: f64,
    /// The render mode for this environment.
    pub render_mode: RenderMode,
}

impl Default for PendulumConfig {
    fn default() -> Self {
        Self {
            g: 10.0,
            render_mode: RenderMode::None,
        }
    }
}

/// Normalize angle to `[-π, π]`.
fn angle_normalize(x: f64) -> f64 {
    ((x + PI) % (2.0 * PI)) - PI
}

/// The inverted pendulum swing-up environment with continuous torque action.
///
/// Based on Andrew Moore's `PhD` Thesis (1990).
/// Reference: <https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf>
///
/// Mirrors [Gymnasium `Pendulum-v1`](https://gymnasium.farama.org/environments/classic_control/pendulum/).
///
/// # Action Space
///
/// [`BoundedSpace`] of shape `[1]`: torque in `[-2.0, 2.0]`.
///
/// # Observation Space
///
/// [`BoundedSpace`] of shape `[3]`:
///
/// | Index | Observation      | Min  | Max |
/// |-------|------------------|------|-----|
/// | 0     | cos(θ)           | −1.0 | 1.0 |
/// | 1     | sin(θ)           | −1.0 | 1.0 |
/// | 2     | Angular velocity | −8.0 | 8.0 |
///
/// # Rewards
///
/// `r = -(θ² + 0.1 * θ̇² + 0.001 * torque²)` where θ is normalized to
/// `[-π, π]`. Maximum reward is 0 (upright, zero velocity, zero torque).
///
/// # Episode End
///
/// The environment never terminates on its own; truncation is handled by a
/// [`TimeLimit`](crate::wrappers::TimeLimit) wrapper (typically 200 steps).
pub struct PendulumEnv {
    action_space: BoundedSpace,
    observation_space: BoundedSpace,

    /// Internal state: `[theta, theta_dot]`.
    state: Option<[f64; 2]>,
    /// Last applied torque (for rendering the action indicator).
    last_u: Option<f64>,
    rng: Rng,
    g: f64,
    render_mode: RenderMode,

    #[cfg(feature = "render")]
    canvas: Option<Canvas>,
    #[cfg(feature = "render")]
    window: Option<RenderWindow>,
}

impl std::fmt::Debug for PendulumEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendulumEnv")
            .field("state", &self.state)
            .field("render_mode", &self.render_mode)
            .finish_non_exhaustive()
    }
}

impl PendulumEnv {
    /// Create a new pendulum environment.
    ///
    /// # Errors
    ///
    /// Returns an error if the observation or action space cannot be constructed.
    pub fn new(config: PendulumConfig) -> Result<Self> {
        #[allow(clippy::cast_possible_truncation)]
        let obs_high = vec![1.0_f32, 1.0, MAX_SPEED as f32];
        let obs_low: Vec<f32> = obs_high.iter().map(|&h| -h).collect();

        #[allow(clippy::cast_possible_truncation)]
        let act_high = vec![MAX_TORQUE as f32];
        let act_low: Vec<f32> = act_high.iter().map(|&h| -h).collect();

        Ok(Self {
            observation_space: BoundedSpace::new(obs_low, obs_high)?,
            action_space: BoundedSpace::new(act_low, act_high)?,
            state: None,
            last_u: None,
            rng: rng::create_rng(None),
            g: config.g,
            render_mode: config.render_mode,
            #[cfg(feature = "render")]
            canvas: None,
            #[cfg(feature = "render")]
            window: None,
        })
    }

    /// Build the `[cos(θ), sin(θ), θ̇]` observation.
    #[allow(clippy::cast_possible_truncation)]
    fn observation(&self) -> Vec<f32> {
        let [theta, theta_dot] = self.state.expect("state must be initialized");
        vec![theta.cos() as f32, theta.sin() as f32, theta_dot as f32]
    }

    /// Render the pendulum scene to the internal canvas.
    ///
    /// Matches Gymnasium's rendering exactly: rotated rectangle rod with
    /// rounded end-caps, proper colors, and correct coordinate transforms
    /// (Gymnasium draws in Y-down then flips; we compute final screen coords
    /// directly).
    #[cfg(feature = "render")]
    #[allow(clippy::cast_possible_truncation)]
    fn render_pixels(&mut self) -> Result<RenderFrame> {
        if self.state.is_none() {
            return Err(Error::ResetNeeded { method: "render" });
        }
        let [theta, _] = self.state.expect("checked above");

        let canvas = self
            .canvas
            .get_or_insert_with(|| Canvas::new(SCREEN_DIM, SCREEN_DIM));

        canvas.clear(tiny_skia::Color::WHITE);

        // Gymnasium: bound=2.2, scale=screen_dim/(bound*2), offset=screen_dim//2
        let bound = 2.2_f32;
        let scale = SCREEN_DIM as f32 / (bound * 2.0);
        let offset = SCREEN_DIM as f32 / 2.0;

        let rod_length = scale; // 1.0 * scale
        let rod_width = 0.2 * scale;
        let rod_color = tiny_skia::Color::from_rgba8(204, 77, 77, 255);

        // Rod rectangle in local coords: extends along +X from 0 to rod_length,
        // half-width rod_width/2 above and below.
        let rw2 = rod_width / 2.0;
        let corners_local: [(f32, f32); 4] = [
            (0.0, -rw2),
            (0.0, rw2),
            (rod_length, rw2),
            (rod_length, -rw2),
        ];

        // Gymnasium rotates by (theta + pi/2) then Y-flips. The combined
        // transform for corner (lx, ly) → screen coords is:
        //   sx = offset - lx*sin(theta) - ly*cos(theta)
        //   sy = offset - lx*cos(theta) + ly*sin(theta)
        let sin_t = (theta as f32).sin();
        let cos_t = (theta as f32).cos();

        let rod_corners: Vec<(f32, f32)> = corners_local
            .iter()
            .map(|&(lx, ly)| {
                let sx = offset - lx.mul_add(sin_t, ly * cos_t);
                let sy = offset - lx.mul_add(cos_t, -(ly * sin_t));
                (sx, sy)
            })
            .collect();

        canvas.fill_polygon(&rod_corners, rod_color);

        // Circle at rod start (pivot end) — cosmetic cap.
        canvas.fill_circle(offset, offset, rw2, rod_color);

        // Circle at rod end.
        let end_x = offset - rod_length * sin_t;
        let end_y = offset - rod_length * cos_t;
        canvas.fill_circle(end_x, end_y, rw2, rod_color);

        // Axle dot (black, small).
        let axle_r = 0.05 * scale;
        canvas.fill_circle(offset, offset, axle_r, tiny_skia::Color::BLACK);

        match self.render_mode {
            RenderMode::Human => {
                let window = self.window.get_or_insert_with(|| {
                    RenderWindow::new(
                        "Pendulum \u{2014} gmgn",
                        SCREEN_DIM as usize,
                        SCREEN_DIM as usize,
                        RENDER_FPS,
                    )
                    .expect("failed to create render window")
                });

                if !window.is_open() {
                    return Ok(RenderFrame::None);
                }

                window.show(canvas)?;
                Ok(RenderFrame::None)
            }
            RenderMode::RgbArray => {
                let rgb = canvas.pixels_rgb();
                Ok(RenderFrame::RgbArray {
                    width: SCREEN_DIM,
                    height: SCREEN_DIM,
                    data: rgb,
                })
            }
            _ => Ok(RenderFrame::None),
        }
    }
}

impl Env for PendulumEnv {
    type Obs = Vec<f32>;
    type Act = Vec<f32>;
    type ObsSpace = BoundedSpace;
    type ActSpace = BoundedSpace;

    fn step(&mut self, action: &Vec<f32>) -> Result<StepResult<Vec<f32>>> {
        if self.state.is_none() {
            return Err(Error::ResetNeeded { method: "step" });
        }

        let [theta, theta_dot] = self.state.expect("checked above");

        // Clip torque to valid range.
        let u = f64::from(action[0]).clamp(-MAX_TORQUE, MAX_TORQUE);
        self.last_u = Some(u);

        // Cost (negative reward).
        let th_norm = angle_normalize(theta);
        let cost = (0.001 * u).mul_add(u, th_norm.mul_add(th_norm, 0.1 * theta_dot * theta_dot));

        // Dynamics (Euler integration).
        let new_theta_dot = (3.0 * self.g / (2.0 * LENGTH))
            .mul_add(theta.sin(), 3.0 / (MASS * LENGTH * LENGTH) * u)
            .mul_add(DT, theta_dot)
            .clamp(-MAX_SPEED, MAX_SPEED);
        let new_theta = theta + new_theta_dot * DT;

        self.state = Some([new_theta, new_theta_dot]);

        Ok(StepResult {
            obs: self.observation(),
            reward: -cost,
            terminated: false,
            truncated: false,
            info: HashMap::new(),
        })
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Vec<f32>>> {
        if let Some(s) = seed {
            self.rng = rng::create_rng(Some(s));
        }

        // Random angle in [-π, π], random velocity in [-1, 1].
        let theta = self.rng.random_range(-PI..PI);
        let theta_dot = self.rng.random_range(-1.0..1.0);
        self.state = Some([theta, theta_dot]);
        self.last_u = None;

        Ok(ResetResult {
            obs: self.observation(),
            info: HashMap::new(),
        })
    }

    fn render(&mut self) -> Result<RenderFrame> {
        match self.render_mode {
            RenderMode::None => Ok(RenderFrame::None),
            RenderMode::Ansi => {
                if self.state.is_none() {
                    return Err(Error::ResetNeeded { method: "render" });
                }
                let [theta, theta_dot] = self.state.expect("checked above");
                Ok(RenderFrame::Ansi(format!(
                    "Pendulum | θ: {theta:+.3} rad | θ̇: {theta_dot:+.3}"
                )))
            }
            #[cfg(feature = "render")]
            RenderMode::Human | RenderMode::RgbArray => self.render_pixels(),
            #[cfg(not(feature = "render"))]
            _ => Err(Error::UnsupportedRenderMode {
                mode: format!("{:?}", self.render_mode),
            }),
        }
    }

    fn observation_space(&self) -> &BoundedSpace {
        &self.observation_space
    }

    fn action_space(&self) -> &BoundedSpace {
        &self.action_space
    }

    fn render_mode(&self) -> &RenderMode {
        &self.render_mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::Space;

    fn make_env() -> PendulumEnv {
        PendulumEnv::new(PendulumConfig::default()).unwrap()
    }

    #[test]
    fn reset_produces_valid_observation() {
        let mut env = make_env();
        let r = env.reset(Some(42)).unwrap();
        assert_eq!(r.obs.len(), 3);
        assert!(env.observation_space().contains(&r.obs));
    }

    #[test]
    fn step_without_reset_errors() {
        let mut env = make_env();
        assert!(env.step(&vec![0.0]).is_err());
    }

    #[test]
    fn step_returns_valid_observation() {
        let mut env = make_env();
        env.reset(Some(42)).unwrap();
        let r = env.step(&vec![1.0]).unwrap();
        assert_eq!(r.obs.len(), 3);
        // cos²θ + sin²θ ≈ 1
        let cos_sq_plus_sin_sq =
            f64::from(r.obs[1]).mul_add(f64::from(r.obs[1]), f64::from(r.obs[0]).powi(2));
        assert!((cos_sq_plus_sin_sq - 1.0).abs() < 1e-5);
    }

    #[test]
    fn never_terminates() {
        let mut env = make_env();
        env.reset(Some(0)).unwrap();
        for _ in 0..500 {
            let r = env.step(&vec![2.0]).unwrap();
            assert!(!r.terminated);
            assert!(!r.truncated);
        }
    }

    #[test]
    fn reward_is_non_positive() {
        let mut env = make_env();
        env.reset(Some(42)).unwrap();
        for _ in 0..100 {
            let r = env.step(&vec![0.5]).unwrap();
            assert!(r.reward <= 0.0);
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let mut e1 = make_env();
        let mut e2 = make_env();

        let r1 = e1.reset(Some(99)).unwrap();
        let r2 = e2.reset(Some(99)).unwrap();
        assert_eq!(r1.obs, r2.obs);

        let s1 = e1.step(&vec![0.5]).unwrap();
        let s2 = e2.step(&vec![0.5]).unwrap();
        assert_eq!(s1.obs, s2.obs);
        assert!((s1.reward - s2.reward).abs() < f64::EPSILON);
    }

    #[test]
    fn action_clipped() {
        let mut env = make_env();
        env.reset(Some(42)).unwrap();
        // Action exceeds bounds — should be silently clipped, no error.
        let r = env.step(&vec![100.0]).unwrap();
        assert_eq!(r.obs.len(), 3);
    }

    #[test]
    fn angle_normalize_works() {
        assert!((angle_normalize(0.0) - 0.0).abs() < 1e-10);
        assert!((angle_normalize(2.0 * PI) - 0.0).abs() < 1e-10);
        assert!((angle_normalize(-PI) - (-PI)).abs() < 1e-10);
    }
}
