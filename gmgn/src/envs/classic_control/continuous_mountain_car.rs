//! Continuous mountain car environment (continuous action).
//!
//! A car is placed stochastically at the bottom of a sinusoidal valley. The
//! action is a continuous force in `[-1, 1]`. The goal is to reach the flag
//! on top of the right hill.
//!
//! Based on Olivier Sigaud's merge of Andrew Moore's `PhD` Thesis (1990) and
//! the Gymnasium `MountainCar` environment.
//!
//! Mirrors [Gymnasium `MountainCarContinuous-v0`](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/).

use std::collections::HashMap;

use crate::env::{Env, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::{Error, Result};
use crate::rng::{self, Rng};
use crate::space::BoundedSpace;

use rand::RngExt as _;

#[cfg(feature = "render")]
use crate::render::{Canvas, RenderWindow};

const MIN_POSITION: f64 = -1.2;
const MAX_POSITION: f64 = 0.6;
const MAX_SPEED: f64 = 0.07;
const GOAL_POSITION: f64 = 0.45;
const POWER: f64 = 0.0015;
const GRAVITY: f64 = 0.0025;

#[cfg(feature = "render")]
const SCREEN_WIDTH: u32 = 600;
#[cfg(feature = "render")]
const SCREEN_HEIGHT: u32 = 400;
#[cfg(feature = "render")]
const RENDER_FPS: usize = 30;
#[cfg(feature = "render")]
const CAR_WIDTH: f32 = 40.0;
#[cfg(feature = "render")]
const CAR_HEIGHT: f32 = 20.0;

/// Configuration for [`ContinuousMountainCarEnv`].
#[derive(Debug, Clone, Copy)]
pub struct ContinuousMountainCarConfig {
    /// Minimum velocity required at the goal to count as solved.
    pub goal_velocity: f64,
    /// The render mode for this environment.
    pub render_mode: RenderMode,
}

impl Default for ContinuousMountainCarConfig {
    fn default() -> Self {
        Self {
            goal_velocity: 0.0,
            render_mode: RenderMode::None,
        }
    }
}

/// The continuous mountain car environment.
///
/// # Action Space
///
/// [`BoundedSpace`] of shape `[1]`: force in `[-1.0, 1.0]`.
///
/// # Observation Space
///
/// [`BoundedSpace`] of shape `[2]`:
///
/// | Index | Observation | Min   | Max  |
/// |-------|-------------|-------|------|
/// | 0     | Position    | −1.2  | 0.6  |
/// | 1     | Velocity    | −0.07 | 0.07 |
///
/// # Rewards
///
/// `−0.1 * action²` per step; +100 bonus when the goal is reached.
///
/// # Episode End
///
/// - **Termination**: position ≥ 0.45 **and** velocity ≥ `goal_velocity`.
/// - **Truncation**: handled externally by a [`TimeLimit`](crate::wrappers::TimeLimit)
///   wrapper (typically 999 steps).
pub struct ContinuousMountainCarEnv {
    action_space: BoundedSpace,
    observation_space: BoundedSpace,

    state: Option<[f64; 2]>,
    rng: Rng,
    goal_velocity: f64,
    render_mode: RenderMode,

    #[cfg(feature = "render")]
    canvas: Option<Canvas>,
    #[cfg(feature = "render")]
    window: Option<RenderWindow>,
}

impl std::fmt::Debug for ContinuousMountainCarEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContinuousMountainCarEnv")
            .field("state", &self.state)
            .field("render_mode", &self.render_mode)
            .finish_non_exhaustive()
    }
}

impl ContinuousMountainCarEnv {
    /// Create a new continuous mountain car environment.
    ///
    /// # Errors
    ///
    /// Returns an error if observation or action space cannot be constructed.
    pub fn new(config: ContinuousMountainCarConfig) -> Result<Self> {
        #[allow(clippy::cast_possible_truncation)]
        let obs_low = vec![MIN_POSITION as f32, -MAX_SPEED as f32];
        #[allow(clippy::cast_possible_truncation)]
        let obs_high = vec![MAX_POSITION as f32, MAX_SPEED as f32];

        let act_low = vec![-1.0_f32];
        let act_high = vec![1.0_f32];

        Ok(Self {
            observation_space: BoundedSpace::new(obs_low, obs_high)?,
            action_space: BoundedSpace::new(act_low, act_high)?,
            state: None,
            rng: rng::create_rng(None),
            goal_velocity: config.goal_velocity,
            render_mode: config.render_mode,
            #[cfg(feature = "render")]
            canvas: None,
            #[cfg(feature = "render")]
            window: None,
        })
    }

    /// Extract the current state as an `f32` observation vector.
    #[allow(clippy::cast_possible_truncation)]
    fn observation(&self) -> Vec<f32> {
        let s = self.state.expect("state must be initialized");
        vec![s[0] as f32, s[1] as f32]
    }

    /// Terrain height function: `sin(3x) * 0.45 + 0.55`.
    #[cfg(feature = "render")]
    fn height(x: f64) -> f64 {
        (3.0 * x).sin().mul_add(0.45, 0.55)
    }

    /// Render the scene to the internal canvas (shares rendering with discrete
    /// version — identical visuals).
    #[cfg(feature = "render")]
    #[allow(clippy::cast_possible_truncation)]
    fn render_pixels(&mut self) -> Result<RenderFrame> {
        if self.state.is_none() {
            return Err(Error::ResetNeeded { method: "render" });
        }
        let [pos, _vel] = self.state.expect("checked above");

        let canvas = self
            .canvas
            .get_or_insert_with(|| Canvas::new(SCREEN_WIDTH, SCREEN_HEIGHT));

        canvas.clear(tiny_skia::Color::WHITE);

        let world_width = MAX_POSITION - MIN_POSITION;
        let scale = f64::from(SCREEN_WIDTH) / world_width;
        let h = f64::from(SCREEN_HEIGHT);

        // Draw terrain polyline.
        let n_points = 100;
        let terrain: Vec<(f32, f32)> = (0..=n_points)
            .map(|i| {
                let x = (MAX_POSITION - MIN_POSITION)
                    .mul_add(f64::from(i) / f64::from(n_points), MIN_POSITION);
                let sx = ((x - MIN_POSITION) * scale) as f32;
                let sy = (h - Self::height(x) * scale) as f32;
                (sx, sy)
            })
            .collect();
        canvas.stroke_polyline(&terrain, 2.0, tiny_skia::Color::BLACK);

        // Car position on the curve (Y-flipped: screen_y = h - world_y).
        let car_world_x = ((pos - MIN_POSITION) * scale) as f32;
        let car_world_y = (Self::height(pos) * scale) as f32;
        let clearance = 10.0_f32;

        // Gymnasium rotates by cos(3*pos).
        let rot_angle = (3.0 * pos).cos() as f32;
        let sin_a = rot_angle.sin();
        let cos_a = rot_angle.cos();
        let hw = CAR_WIDTH / 2.0;
        let hh = CAR_HEIGHT;

        let corners_local: [(f32, f32); 4] =
            [(-hw, 0.0), (-hw, hh), (hw, hh), (hw, 0.0)];

        let car_corners: Vec<(f32, f32)> = corners_local
            .iter()
            .map(|&(lx, ly)| {
                let rx = lx.mul_add(cos_a, -(ly * sin_a)) + car_world_x;
                let ry = h as f32
                    - (lx.mul_add(sin_a, ly * cos_a) + clearance + car_world_y);
                (rx, ry)
            })
            .collect();

        canvas.fill_polygon(&car_corners, tiny_skia::Color::BLACK);

        // Wheels.
        let wheel_radius = CAR_HEIGHT / 2.5;
        let wheel_color = tiny_skia::Color::from_rgba8(128, 128, 128, 255);
        for &woff in &[CAR_WIDTH / 4.0, -(CAR_WIDTH / 4.0)] {
            let wx = woff.mul_add(cos_a, car_world_x);
            let wy = h as f32 - (woff.mul_add(sin_a, clearance + car_world_y));
            canvas.fill_circle(wx, wy, wheel_radius, wheel_color);
        }

        // Goal flag.
        let flag_x = ((GOAL_POSITION - MIN_POSITION) * scale) as f32;
        let flag_y1 = Self::height(GOAL_POSITION).mul_add(-scale, h) as f32;
        let flag_y2 = flag_y1 - 50.0;
        canvas.stroke_line(flag_x, flag_y1, flag_x, flag_y2, 2.0, tiny_skia::Color::BLACK);

        let flag_color = tiny_skia::Color::from_rgba8(204, 204, 0, 255);
        let flag_verts = vec![
            (flag_x, flag_y2),
            (flag_x, flag_y2 + 10.0),
            (flag_x + 25.0, flag_y2 + 5.0),
        ];
        canvas.fill_polygon(&flag_verts, flag_color);

        match self.render_mode {
            RenderMode::Human => {
                let window = self.window.get_or_insert_with(|| {
                    RenderWindow::new(
                        "MountainCarContinuous \u{2014} gmgn",
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

impl Env for ContinuousMountainCarEnv {
    type Obs = Vec<f32>;
    type Act = Vec<f32>;
    type ObsSpace = BoundedSpace;
    type ActSpace = BoundedSpace;

    fn step(&mut self, action: &Vec<f32>) -> Result<StepResult<Vec<f32>>> {
        if self.state.is_none() {
            return Err(Error::ResetNeeded { method: "step" });
        }

        let [position, velocity] = self.state.expect("checked above");

        // Clip force to [-1, 1].
        let force = f64::from(action[0]).clamp(-1.0, 1.0);

        // Transition dynamics.
        let velocity = (force * POWER - GRAVITY * (3.0 * position).cos() + velocity)
            .clamp(-MAX_SPEED, MAX_SPEED);

        let position = (position + velocity).clamp(MIN_POSITION, MAX_POSITION);

        // Inelastic collision at the left wall.
        if position <= MIN_POSITION && velocity < 0.0 {
            self.state = Some([position, 0.0]);
        } else {
            self.state = Some([position, velocity]);
        }

        let velocity = self.state.expect("set above")[1];
        let terminated = position >= GOAL_POSITION && velocity >= self.goal_velocity;

        // Reward: -0.1 * action^2, +100 on termination.
        let reward = (0.1 * force).mul_add(-force, if terminated { 100.0 } else { 0.0 });

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

        // Position uniform in [-0.6, -0.4], velocity = 0.
        let position = self.rng.random_range(-0.6..=-0.4);
        self.state = Some([position, 0.0]);

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
                let [pos, vel] = self.state.expect("checked above");
                Ok(RenderFrame::Ansi(format!(
                    "MountainCarContinuous | pos: {pos:+.4} | vel: {vel:+.4}"
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

    fn make_env() -> ContinuousMountainCarEnv {
        ContinuousMountainCarEnv::new(ContinuousMountainCarConfig::default()).unwrap()
    }

    #[test]
    fn reset_produces_valid_observation() {
        let mut env = make_env();
        let r = env.reset(Some(42)).unwrap();
        assert_eq!(r.obs.len(), 2);
        assert!(env.observation_space().contains(&r.obs));
        assert!((r.obs[1] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn step_without_reset_errors() {
        let mut env = make_env();
        assert!(env.step(&vec![0.0]).is_err());
    }

    #[test]
    fn reward_includes_action_penalty() {
        let mut env = make_env();
        env.reset(Some(42)).unwrap();
        // Zero action → reward = -0.1 * 0^2 = 0
        let r0 = env.step(&vec![0.0]).unwrap();
        assert!((r0.reward - 0.0).abs() < 1e-10);
        // Full action → reward = -0.1 * 1^2 = -0.1
        env.reset(Some(42)).unwrap();
        let r1 = env.step(&vec![1.0]).unwrap();
        assert!((r1.reward - (-0.1)).abs() < 1e-10);
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
    fn action_is_clipped() {
        let mut env = make_env();
        env.reset(Some(42)).unwrap();
        // Action exceeds bounds — should be silently clipped.
        let r = env.step(&vec![100.0]).unwrap();
        assert_eq!(r.obs.len(), 2);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn position_clipped_to_bounds() {
        let mut env = make_env();
        env.reset(Some(0)).unwrap();
        for _ in 0..1000 {
            let r = env.step(&vec![-1.0]).unwrap();
            assert!(r.obs[0] >= MIN_POSITION as f32);
        }
    }
}
