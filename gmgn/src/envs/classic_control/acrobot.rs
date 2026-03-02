//! Two-link acrobot swing-up environment.
//!
//! The system consists of two links connected linearly to form a chain, with
//! one end fixed. The joint between the two links is actuated. The goal is to
//! swing the free end above a target height.
//!
//! Based on Sutton's "Generalization in Reinforcement Learning" (`NeurIPS` 1995).
//!
//! Mirrors [Gymnasium `Acrobot-v1`](https://gymnasium.farama.org/environments/classic_control/acrobot/).

use std::collections::HashMap;
use std::f64::consts::PI;

use rand::RngExt as _;

use crate::env::{Env, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::{Error, Result};
#[cfg(feature = "render")]
use crate::render::{Canvas, RenderWindow};
use crate::rng::{self, Rng};
use crate::space::{BoundedSpace, Discrete, Space};

const LINK_LENGTH_1: f64 = 1.0;
const LINK_LENGTH_2: f64 = 1.0;
const LINK_MASS_1: f64 = 1.0;
const LINK_MASS_2: f64 = 1.0;
const LINK_COM_POS_1: f64 = 0.5;
const LINK_COM_POS_2: f64 = 0.5;
const LINK_MOI: f64 = 1.0;
const GRAVITY: f64 = 9.8;
const DT: f64 = 0.2;
const MAX_VEL_1: f64 = 4.0 * PI;
const MAX_VEL_2: f64 = 9.0 * PI;

#[cfg(feature = "render")]
const SCREEN_DIM: u32 = 500;
#[cfg(feature = "render")]
const RENDER_FPS: usize = 15;

/// Available torques mapped from discrete actions {0, 1, 2}.
const AVAIL_TORQUE: [f64; 3] = [-1.0, 0.0, 1.0];

/// Configuration for [`AcrobotEnv`].
#[derive(Debug, Clone, Copy)]
pub struct AcrobotConfig {
    /// The render mode for this environment.
    pub render_mode: RenderMode,
}

impl Default for AcrobotConfig {
    fn default() -> Self {
        Self {
            render_mode: RenderMode::None,
        }
    }
}

/// Wrap angle to `[-π, π]`.
fn wrap(x: f64, lo: f64, hi: f64) -> f64 {
    let range = hi - lo;
    ((x - lo) % range + range) % range + lo
}

/// Clamp value to `[lo, hi]`.
const fn bound(x: f64, lo: f64, hi: f64) -> f64 {
    x.clamp(lo, hi)
}

/// Compute the state derivatives for the acrobot (book dynamics).
///
/// `s_aug = [theta1, theta2, dtheta1, dtheta2, torque]`
fn dsdt(s_aug: &[f64; 5]) -> [f64; 5] {
    let [theta1, theta2, dtheta1, dtheta2, a] = *s_aug;

    let m1 = LINK_MASS_1;
    let m2 = LINK_MASS_2;
    let l1 = LINK_LENGTH_1;
    let lc1 = LINK_COM_POS_1;
    let lc2 = LINK_COM_POS_2;
    let i1 = LINK_MOI;
    let i2 = LINK_MOI;
    let g = GRAVITY;

    let d1 = (m1 * lc1).mul_add(
        lc1,
        m2 * (2.0 * l1 * lc2).mul_add(theta2.cos(), l1.mul_add(l1, lc2 * lc2)),
    ) + i1
        + i2;
    let d2 = m2.mul_add(lc2.mul_add(lc2, l1 * lc2 * theta2.cos()), i2);
    let phi2 = m2 * lc2 * g * (theta1 + theta2 - PI / 2.0).cos();
    let phi1 = (m1.mul_add(lc1, m2 * l1) * g).mul_add(
        (theta1 - PI / 2.0).cos(),
        (-m2 * l1 * lc2 * dtheta2 * dtheta2).mul_add(
            theta2.sin(),
            -(2.0 * m2 * l1 * lc2 * dtheta2 * dtheta1 * theta2.sin()),
        ),
    ) + phi2;

    // Book dynamics (Sutton & Barto).
    let ddtheta2 = ((m2 * l1 * lc2 * dtheta1 * dtheta1)
        .mul_add(-theta2.sin(), (d2 / d1).mul_add(phi1, a))
        - phi2)
        / ((m2 * lc2).mul_add(lc2, i2) - d2 * d2 / d1);
    let ddtheta1 = -d2.mul_add(ddtheta2, phi1) / d1;

    [dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0]
}

/// Fourth-order Runge-Kutta integrator for the acrobot dynamics.
fn rk4(state_aug: &[f64; 5], dt: f64) -> [f64; 5] {
    // Single step RK4 over interval [0, dt].
    let k1 = dsdt(state_aug);

    let mut s2 = [0.0; 5];
    for i in 0..5 {
        s2[i] = (0.5 * dt).mul_add(k1[i], state_aug[i]);
    }
    let k2 = dsdt(&s2);

    let mut s3 = [0.0; 5];
    for i in 0..5 {
        s3[i] = (0.5 * dt).mul_add(k2[i], state_aug[i]);
    }
    let k3 = dsdt(&s3);

    let mut s4 = [0.0; 5];
    for i in 0..5 {
        s4[i] = dt.mul_add(k3[i], state_aug[i]);
    }
    let k4 = dsdt(&s4);

    let mut result = [0.0; 5];
    for i in 0..5 {
        result[i] = (dt / 6.0).mul_add(
            2.0f64.mul_add(k3[i], 2.0f64.mul_add(k2[i], k1[i])) + k4[i],
            state_aug[i],
        );
    }
    result
}

/// The Acrobot swing-up environment with discrete actions.
///
/// # Action Space
///
/// [`Discrete(3)`](Discrete): apply torque −1 (0), 0 (1), or +1 (2).
///
/// # Observation Space
///
/// [`BoundedSpace`] of shape `[6]`:
///
/// | Index | Observation          | Min    | Max    |
/// |-------|----------------------|--------|--------|
/// | 0     | cos(θ₁)              | −1     | 1      |
/// | 1     | sin(θ₁)              | −1     | 1      |
/// | 2     | cos(θ₂)              | −1     | 1      |
/// | 3     | sin(θ₂)              | −1     | 1      |
/// | 4     | Angular vel θ₁       | −4π    | 4π     |
/// | 5     | Angular vel θ₂       | −9π    | 9π     |
///
/// # Rewards
///
/// −1 per step until termination; 0 on the terminal step.
///
/// # Episode End
///
/// - **Termination**: `−cos(θ₁) − cos(θ₂ + θ₁) > 1.0`.
/// - **Truncation**: handled externally by a [`TimeLimit`](crate::wrappers::TimeLimit)
///   wrapper (typically 500 steps).
pub struct AcrobotEnv {
    action_space: Discrete,
    observation_space: BoundedSpace,

    /// Internal state: `[theta1, theta2, dtheta1, dtheta2]`.
    state: Option<[f64; 4]>,
    rng: Rng,
    render_mode: RenderMode,

    #[cfg(feature = "render")]
    canvas: Option<Canvas>,
    #[cfg(feature = "render")]
    window: Option<RenderWindow>,
}

impl std::fmt::Debug for AcrobotEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcrobotEnv")
            .field("state", &self.state)
            .field("render_mode", &self.render_mode)
            .finish_non_exhaustive()
    }
}

impl AcrobotEnv {
    /// Create a new acrobot environment.
    ///
    /// # Errors
    ///
    /// Returns an error if the observation space cannot be constructed.
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(config: AcrobotConfig) -> Result<Self> {
        let obs_high = vec![1.0_f32, 1.0, 1.0, 1.0, MAX_VEL_1 as f32, MAX_VEL_2 as f32];
        let obs_low: Vec<f32> = obs_high.iter().map(|&h| -h).collect();

        Ok(Self {
            action_space: Discrete::new(3),
            observation_space: BoundedSpace::new(obs_low, obs_high)?,
            state: None,
            rng: rng::create_rng(None),
            render_mode: config.render_mode,
            #[cfg(feature = "render")]
            canvas: None,
            #[cfg(feature = "render")]
            window: None,
        })
    }

    /// Build the 6-element observation from internal state.
    #[allow(clippy::cast_possible_truncation)]
    fn observation(&self) -> Vec<f32> {
        let [t1, t2, dt1, dt2] = self.state.expect("state must be initialized");
        vec![
            t1.cos() as f32,
            t1.sin() as f32,
            t2.cos() as f32,
            t2.sin() as f32,
            dt1 as f32,
            dt2 as f32,
        ]
    }

    /// Check the termination condition.
    fn is_terminal(&self) -> bool {
        let [t1, t2, _, _] = self.state.expect("state must be initialized");
        -t1.cos() - (t2 + t1).cos() > 1.0
    }

    /// Render the acrobot scene to the internal canvas.
    ///
    /// Matches Gymnasium's rendering: rotated rectangle links with joint
    /// circles, proper colors (cyan links, yellow joints), and correct
    /// coordinate transforms accounting for Gymnasium's Y-flip.
    #[cfg(feature = "render")]
    #[allow(clippy::cast_possible_truncation)]
    fn render_pixels(&mut self) -> Result<RenderFrame> {
        if self.state.is_none() {
            return Err(Error::ResetNeeded { method: "render" });
        }
        let [t1, t2, _, _] = self.state.expect("checked above");

        let canvas = self
            .canvas
            .get_or_insert_with(|| Canvas::new(SCREEN_DIM, SCREEN_DIM));

        canvas.clear(tiny_skia::Color::WHITE);

        let bound = (LINK_LENGTH_1 + LINK_LENGTH_2 + 0.2) as f32;
        let scale = SCREEN_DIM as f32 / (bound * 2.0);
        let offset = SCREEN_DIM as f32 / 2.0;
        let link_width = 0.1 * scale; // half-width for link rectangles

        let link_color = tiny_skia::Color::from_rgba8(0, 204, 204, 255);
        let joint_color = tiny_skia::Color::from_rgba8(204, 204, 0, 255);
        let joint_radius = 0.1 * scale;

        // Joint positions (after Gymnasium's Y-flip transform):
        // Pivot at (offset, offset).
        // Elbow: (L1*sin(t1)*scale + offset, L1*cos(t1)*scale + offset).
        let p1_x = (LINK_LENGTH_1 as f32 * (t1 as f32).sin()).mul_add(scale, offset);
        let p1_y = (LINK_LENGTH_1 as f32 * (t1 as f32).cos()).mul_add(scale, offset);

        // Tip: ((L1*sin(t1)+L2*sin(t1+t2))*scale + offset, ...).
        let p2_x = (LINK_LENGTH_1 as f32)
            .mul_add(
                (t1 as f32).sin(),
                LINK_LENGTH_2 as f32 * ((t1 + t2) as f32).sin(),
            )
            .mul_add(scale, offset);
        let p2_y = (LINK_LENGTH_1 as f32)
            .mul_add(
                (t1 as f32).cos(),
                LINK_LENGTH_2 as f32 * ((t1 + t2) as f32).cos(),
            )
            .mul_add(scale, offset);

        // Target height line at y = offset - scale (1 link-length above pivot).
        let target_y = offset - scale;
        canvas.stroke_line(
            0.0,
            target_y,
            SCREEN_DIM as f32,
            target_y,
            1.0,
            tiny_skia::Color::BLACK,
        );

        // Helper: draw a rotated rectangle link from joint (jx, jy) with
        // rotation angle theta (Gymnasium uses theta - pi/2 as rotate_rad arg).
        // After the combined rotate + Y-flip transform, for local corner (lx, ly):
        //   sx = lx*sin(theta) + ly*cos(theta) + jx
        //   sy = lx*cos(theta) - ly*sin(theta) + jy
        let draw_link = |canvas: &mut Canvas, jx: f32, jy: f32, theta: f64, llen: f32| {
            let sin_t = (theta as f32).sin();
            let cos_t = (theta as f32).cos();
            let lw = link_width;
            // Rectangle: (0, -lw) to (llen, lw)
            let corners: [(f32, f32); 4] = [(0.0, -lw), (0.0, lw), (llen, lw), (llen, -lw)];
            let transformed: Vec<(f32, f32)> = corners
                .iter()
                .map(|&(lx, ly)| {
                    let sx = lx.mul_add(sin_t, ly * cos_t) + jx;
                    let sy = lx.mul_add(cos_t, -(ly * sin_t)) + jy;
                    (sx, sy)
                })
                .collect();
            canvas.fill_polygon(&transformed, link_color);
        };

        // Link 1: from pivot, rotation angle = t1
        let l1_len = LINK_LENGTH_1 as f32 * scale;
        draw_link(canvas, offset, offset, t1, l1_len);

        // Link 2: from elbow, rotation angle = t1 + t2
        let l2_len = LINK_LENGTH_2 as f32 * scale;
        draw_link(canvas, p1_x, p1_y, t1 + t2, l2_len);

        // Joint circles (yellow).
        canvas.fill_circle(offset, offset, joint_radius, joint_color);
        canvas.fill_circle(p1_x, p1_y, joint_radius, joint_color);

        // Tip indicator (small circle).
        canvas.fill_circle(p2_x, p2_y, joint_radius * 0.5, joint_color);

        match self.render_mode {
            RenderMode::Human => {
                let window = self.window.get_or_insert_with(|| {
                    RenderWindow::new(
                        "Acrobot \u{2014} gmgn",
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

impl Env for AcrobotEnv {
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
                reason: format!("expected 0, 1, or 2, got {action}"),
            });
        }

        let [t1, t2, dt1, dt2] = self.state.expect("checked above");
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        // Action already validated.
        let torque = AVAIL_TORQUE[*action as usize];

        // Augmented state for the integrator.
        let s_aug = [t1, t2, dt1, dt2, torque];
        let ns = rk4(&s_aug, DT);

        let new_t1 = wrap(ns[0], -PI, PI);
        let new_t2 = wrap(ns[1], -PI, PI);
        let new_dt1 = bound(ns[2], -MAX_VEL_1, MAX_VEL_1);
        let new_dt2 = bound(ns[3], -MAX_VEL_2, MAX_VEL_2);

        self.state = Some([new_t1, new_t2, new_dt1, new_dt2]);

        let terminated = self.is_terminal();
        let reward = if terminated { 0.0 } else { -1.0 };

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

        // All four state variables uniform in [-0.1, 0.1].
        let state: [f64; 4] = std::array::from_fn(|_| self.rng.random_range(-0.1..0.1));
        self.state = Some(state);

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
                let [t1, t2, dt1, dt2] = self.state.expect("checked above");
                Ok(RenderFrame::Ansi(format!(
                    "Acrobot | θ₁: {t1:+.3} | θ₂: {t2:+.3} | ω₁: {dt1:+.3} | ω₂: {dt2:+.3}"
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

    fn make_env() -> AcrobotEnv {
        AcrobotEnv::new(AcrobotConfig::default()).unwrap()
    }

    #[test]
    fn reset_produces_valid_observation() {
        let mut env = make_env();
        let r = env.reset(Some(42)).unwrap();
        assert_eq!(r.obs.len(), 6);
        assert!(env.observation_space().contains(&r.obs));
    }

    #[test]
    fn step_without_reset_errors() {
        let mut env = make_env();
        assert!(env.step(&0).is_err());
    }

    #[test]
    fn step_invalid_action_errors() {
        let mut env = make_env();
        env.reset(Some(0)).unwrap();
        assert!(env.step(&5).is_err());
    }

    #[test]
    fn step_returns_valid_observation() {
        let mut env = make_env();
        env.reset(Some(42)).unwrap();
        let r = env.step(&2).unwrap();
        assert_eq!(r.obs.len(), 6);
        // cos² + sin² ≈ 1 for both angles.
        let c1 = f64::from(r.obs[1]).mul_add(f64::from(r.obs[1]), f64::from(r.obs[0]).powi(2));
        let c2 = f64::from(r.obs[3]).mul_add(f64::from(r.obs[3]), f64::from(r.obs[2]).powi(2));
        assert!((c1 - 1.0).abs() < 1e-5);
        assert!((c2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn reward_is_negative_one_before_termination() {
        let mut env = make_env();
        env.reset(Some(42)).unwrap();
        let r = env.step(&1).unwrap();
        if !r.terminated {
            assert!((r.reward - (-1.0)).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let mut e1 = make_env();
        let mut e2 = make_env();

        let r1 = e1.reset(Some(99)).unwrap();
        let r2 = e2.reset(Some(99)).unwrap();
        assert_eq!(r1.obs, r2.obs);

        let s1 = e1.step(&2).unwrap();
        let s2 = e2.step(&2).unwrap();
        assert_eq!(s1.obs, s2.obs);
        assert!((s1.reward - s2.reward).abs() < f64::EPSILON);
    }

    #[test]
    fn wrap_angle_works() {
        assert!((wrap(0.0, -PI, PI) - 0.0).abs() < 1e-10);
        assert!((wrap(2.0 * PI, -PI, PI) - 0.0).abs() < 1e-10);
    }
}
