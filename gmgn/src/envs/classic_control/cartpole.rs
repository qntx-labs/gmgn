//! Classic cart-pole balancing environment.
//!
//! A pole is attached by an un-actuated joint to a cart that moves along a
//! frictionless track. The goal is to keep the pole balanced by applying
//! forces to the left or right on the cart.
//!
//! Based on the description by Barto, Sutton, and Anderson (1983).
//! Reference: <https://ieeexplore.ieee.org/document/6313077>

use std::collections::HashMap;

use crate::env::{Env, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::{Error, Result};
use crate::rng::{self, Rng};
use crate::space::{BoundedSpace, Discrete, Space};

use rand::RngExt as _;

#[cfg(feature = "render")]
use crate::render::{Canvas, RenderWindow};

const GRAVITY: f64 = 9.8;
const CART_MASS: f64 = 1.0;
const POLE_MASS: f64 = 0.1;
const TOTAL_MASS: f64 = CART_MASS + POLE_MASS;
const POLE_HALF_LENGTH: f64 = 0.5;
const POLE_MASS_LENGTH: f64 = POLE_MASS * POLE_HALF_LENGTH;
const FORCE_MAG: f64 = 10.0;
const TAU: f64 = 0.02;

const THETA_THRESHOLD_RAD: f64 = 12.0 * 2.0 * std::f64::consts::PI / 360.0;
const X_THRESHOLD: f64 = 2.4;

const SCREEN_WIDTH: u32 = 600;
const SCREEN_HEIGHT: u32 = 400;
const RENDER_FPS: usize = 50;
const CART_WIDTH: f32 = 50.0;
const CART_HEIGHT: f32 = 30.0;
const POLE_WIDTH: f32 = 10.0;

#[allow(clippy::cast_possible_truncation)] // Intentional: physics uses f64, obs returns f32.
const OBS_HIGH: [f32; 4] = [
    (X_THRESHOLD * 2.0) as f32,
    f32::INFINITY,
    (THETA_THRESHOLD_RAD * 2.0) as f32,
    f32::INFINITY,
];

/// Configuration for the [`CartPoleEnv`].
#[derive(Debug, Clone, Copy)]
pub struct CartPoleConfig {
    /// If `true`, use the Sutton & Barto reward scheme:
    /// 0 for non-terminal steps, −1 at termination.
    pub sutton_barto_reward: bool,
    /// The render mode for this environment.
    pub render_mode: RenderMode,
}

impl Default for CartPoleConfig {
    fn default() -> Self {
        Self {
            sutton_barto_reward: false,
            render_mode: RenderMode::None,
        }
    }
}

/// The classic cart-pole balancing environment.
///
/// # Action Space
///
/// [`Discrete(2)`](Discrete): push cart left (0) or right (1).
///
/// # Observation Space
///
/// [`BoundedSpace`] of shape `[4]`:
///
/// | Index | Observation           | Min     | Max    |
/// |-------|-----------------------|---------|--------|
/// | 0     | Cart position         | −4.8    | 4.8    |
/// | 1     | Cart velocity         | −∞      | ∞      |
/// | 2     | Pole angle (rad)      | −0.418  | 0.418  |
/// | 3     | Pole angular velocity | −∞      | ∞      |
///
/// # Rewards
///
/// By default, +1 for every step (including the terminal step).
/// With `sutton_barto_reward`, 0 for non-terminal steps and −1 at termination.
///
/// # Episode End
///
/// - **Termination**: pole angle > ±12° or cart position > ±2.4.
/// - **Truncation**: handled externally by a `TimeLimit` wrapper.
pub struct CartPoleEnv {
    // Spaces
    action_space: Discrete,
    observation_space: BoundedSpace,

    // State
    state: Option<[f64; 4]>,
    rng: Rng,
    steps_beyond_terminated: Option<u64>,

    // Configuration
    sutton_barto_reward: bool,
    render_mode: RenderMode,

    // Rendering resources (lazily initialized)
    #[cfg(feature = "render")]
    canvas: Option<Canvas>,
    #[cfg(feature = "render")]
    window: Option<RenderWindow>,
}

impl std::fmt::Debug for CartPoleEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CartPoleEnv")
            .field("state", &self.state)
            .field("render_mode", &self.render_mode)
            .finish_non_exhaustive()
    }
}

impl CartPoleEnv {
    /// Create a new cart-pole environment with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the observation space cannot be constructed.
    pub fn new(config: CartPoleConfig) -> Result<Self> {
        let obs_low: Vec<f32> = OBS_HIGH.iter().map(|&h| -h).collect();
        let obs_high: Vec<f32> = OBS_HIGH.to_vec();

        Ok(Self {
            action_space: Discrete::new(2),
            observation_space: BoundedSpace::new(obs_low, obs_high)?,
            state: None,
            rng: rng::create_rng(None),
            steps_beyond_terminated: None,
            sutton_barto_reward: config.sutton_barto_reward,
            render_mode: config.render_mode,
            #[cfg(feature = "render")]
            canvas: None,
            #[cfg(feature = "render")]
            window: None,
        })
    }

    /// Extract the current state as an `f32` observation vector.
    #[allow(clippy::cast_possible_truncation)] // Intentional: physics f64 → observation f32.
    fn observation(&self) -> Vec<f32> {
        self.state
            .expect("state must be initialized")
            .iter()
            .map(|&v| v as f32)
            .collect()
    }

    /// Render the cart-pole scene to the internal canvas and optionally display it.
    ///
    /// Draws: white background, black track line, black cart rectangle,
    /// brown pole (rotated), blue axle circle — matching Gymnasium's visual output.
    #[cfg(feature = "render")]
    #[allow(clippy::cast_possible_truncation)]
    fn render_pixels(&mut self) -> Result<RenderFrame> {
        if self.state.is_none() {
            return Err(Error::ResetNeeded { method: "render" });
        }
        let [x, _, theta, _] = self.state.expect("checked above");

        // Lazily initialize canvas.
        let canvas = self
            .canvas
            .get_or_insert_with(|| Canvas::new(SCREEN_WIDTH, SCREEN_HEIGHT));

        // Coordinate system: Gymnasium uses Y-up (flip at end), we draw Y-down.
        // We'll use the same layout as Gymnasium but in screen coords directly.
        let world_width = X_THRESHOLD * 2.0;
        let scale = f64::from(SCREEN_WIDTH) / world_width;
        let cart_y = SCREEN_HEIGHT as f32 - 100.0; // cart vertical position from bottom
        let cart_x = x.mul_add(scale, f64::from(SCREEN_WIDTH) / 2.0) as f32;
        let pole_len = (scale * (2.0 * POLE_HALF_LENGTH)) as f32;
        let axle_offset = CART_HEIGHT / 4.0;

        canvas.clear(tiny_skia::Color::WHITE);
        canvas.hline(cart_y, 2.0, tiny_skia::Color::BLACK);
        let cart_left = cart_x - CART_WIDTH / 2.0;
        let cart_top = cart_y - CART_HEIGHT / 2.0;
        canvas.fill_rect(
            cart_left,
            cart_top,
            CART_WIDTH,
            CART_HEIGHT,
            tiny_skia::Color::BLACK,
        );

        // Pole is drawn from axle point upward, rotated by theta.
        let axle_y = cart_y - axle_offset;
        let half_pw = POLE_WIDTH / 2.0;

        // Unrotated pole corners relative to axle (pointing up = negative Y in screen coords).
        let corners: [(f32, f32); 4] = [
            (-half_pw, 0.0),
            (-half_pw, -pole_len),
            (half_pw, -pole_len),
            (half_pw, 0.0),
        ];

        // Rotate each corner by -theta (screen Y is down, so negate).
        let sin_t = (-theta as f32).sin();
        let cos_t = (-theta as f32).cos();
        let rotated: Vec<(f32, f32)> = corners
            .iter()
            .map(|&(lx, ly)| {
                let rx = lx.mul_add(cos_t, -(ly * sin_t)) + cart_x;
                let ry = lx.mul_add(sin_t, ly * cos_t) + axle_y;
                (rx, ry)
            })
            .collect();

        let pole_color = tiny_skia::Color::from_rgba8(202, 152, 101, 255);
        canvas.fill_polygon(&rotated, pole_color);

        let axle_color = tiny_skia::Color::from_rgba8(129, 132, 203, 255);
        canvas.fill_circle(cart_x, axle_y, half_pw, axle_color);

        match self.render_mode {
            RenderMode::Human => {
                let window = self.window.get_or_insert_with(|| {
                    RenderWindow::new(
                        "CartPole — gmgn",
                        SCREEN_WIDTH as usize,
                        SCREEN_HEIGHT as usize,
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
                    width: SCREEN_WIDTH,
                    height: SCREEN_HEIGHT,
                    data: rgb,
                })
            }
            _ => Ok(RenderFrame::None),
        }
    }
}

impl Env for CartPoleEnv {
    type Obs = Vec<f32>;
    type Act = i64;
    type ObsSpace = BoundedSpace;
    type ActSpace = Discrete;

    fn step(&mut self, action: &i64) -> Result<StepResult<Vec<f32>>> {
        if self.state.is_none() {
            return Err(Error::ResetNeeded { method: "step" });
        }

        if !self.action_space.contains(action) {
            return Err(Error::InvalidAction {
                reason: format!("expected 0 or 1, got {action}"),
            });
        }

        let [x, x_dot, theta, theta_dot] = self.state.expect("checked above");
        let force = if *action == 1 { FORCE_MAG } else { -FORCE_MAG };

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // Physics equations (Euler integration).
        // Reference: https://coneural.org/florian/papers/05_cart_pole.pdf
        let temp =
            (POLE_MASS_LENGTH * theta_dot * theta_dot).mul_add(sin_theta, force) / TOTAL_MASS;
        let theta_acc = GRAVITY.mul_add(sin_theta, -(cos_theta * temp))
            / (POLE_HALF_LENGTH * (4.0 / 3.0 - POLE_MASS * cos_theta * cos_theta / TOTAL_MASS));
        let x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

        let x = TAU.mul_add(x_dot, x);
        let x_dot = TAU.mul_add(x_acc, x_dot);
        let theta = TAU.mul_add(theta_dot, theta);
        let theta_dot = TAU.mul_add(theta_acc, theta_dot);

        self.state = Some([x, x_dot, theta, theta_dot]);

        let terminated = !(-X_THRESHOLD..=X_THRESHOLD).contains(&x)
            || !(-THETA_THRESHOLD_RAD..=THETA_THRESHOLD_RAD).contains(&theta);

        let reward = if !terminated {
            if self.sutton_barto_reward { 0.0 } else { 1.0 }
        } else if self.steps_beyond_terminated.is_none() {
            self.steps_beyond_terminated = Some(0);
            if self.sutton_barto_reward { -1.0 } else { 1.0 }
        } else {
            self.steps_beyond_terminated =
                Some(self.steps_beyond_terminated.expect("checked above") + 1);
            if self.sutton_barto_reward { -1.0 } else { 0.0 }
        };

        Ok(StepResult {
            obs: self.observation(),
            reward,
            terminated,
            truncated: false,
            info: HashMap::new(),
        })
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Vec<f32>>> {
        if let Some(s) = seed {
            self.rng = rng::create_rng(Some(s));
        }

        // Initialize state uniformly in [-0.05, 0.05].
        let state: [f64; 4] = std::array::from_fn(|_| self.rng.random_range(-0.05..0.05));
        self.state = Some(state);
        self.steps_beyond_terminated = None;

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
                let [x, _, theta, _] = self.state.expect("checked above");
                Ok(RenderFrame::Ansi(format!(
                    "CartPole | x: {x:+.3} | θ: {theta:+.3} rad"
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

    fn action_space(&self) -> &Discrete {
        &self.action_space
    }

    fn render_mode(&self) -> &RenderMode {
        &self.render_mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_env() -> CartPoleEnv {
        CartPoleEnv::new(CartPoleConfig::default()).unwrap()
    }

    #[test]
    fn reset_produces_valid_observation() {
        let mut env = make_env();
        let result = env.reset(Some(42)).unwrap();
        assert_eq!(result.obs.len(), 4);
        assert!(env.observation_space().contains(&result.obs));
    }

    #[test]
    fn step_without_reset_returns_error() {
        let mut env = make_env();
        let result = env.step(&0);
        assert!(result.is_err());
    }

    #[test]
    fn step_with_invalid_action_returns_error() {
        let mut env = make_env();
        env.reset(Some(42)).unwrap();
        let result = env.step(&5);
        assert!(result.is_err());
    }

    #[test]
    fn step_returns_valid_observation() {
        let mut env = make_env();
        env.reset(Some(42)).unwrap();
        let result = env.step(&1).unwrap();
        assert_eq!(result.obs.len(), 4);
        assert!(!result.truncated);
    }

    #[test]
    fn episode_terminates() {
        let mut env = make_env();
        env.reset(Some(0)).unwrap();
        let mut terminated = false;
        // Push the cart in one direction until termination.
        for _ in 0..500 {
            let result = env.step(&1).unwrap();
            if result.terminated {
                terminated = true;
                break;
            }
        }
        assert!(terminated, "episode should terminate within 500 steps");
    }

    #[test]
    fn deterministic_with_seed() {
        let mut env1 = make_env();
        let mut env2 = make_env();

        let r1 = env1.reset(Some(123)).unwrap();
        let r2 = env2.reset(Some(123)).unwrap();
        assert_eq!(r1.obs, r2.obs);

        let s1 = env1.step(&0).unwrap();
        let s2 = env2.step(&0).unwrap();
        assert_eq!(s1.obs, s2.obs);
        assert!((s1.reward - s2.reward).abs() < f64::EPSILON);
    }

    #[test]
    fn sutton_barto_reward_scheme() {
        let mut env = CartPoleEnv::new(CartPoleConfig {
            sutton_barto_reward: true,
            ..CartPoleConfig::default()
        })
        .unwrap();
        env.reset(Some(42)).unwrap();

        let result = env.step(&0).unwrap();
        if !result.terminated {
            assert!((result.reward - 0.0).abs() < f64::EPSILON);
        }
    }
}
