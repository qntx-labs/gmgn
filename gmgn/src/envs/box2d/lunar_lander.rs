//! Lunar Lander environment using `Box2D` physics.
//!
//! A classic rocket trajectory optimization problem where the agent must land
//! a spacecraft on a designated pad. Supports both discrete (4 actions) and
//! continuous (2D throttle) action spaces.
//!
//! Mirrors [Gymnasium `LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

// box2d-rs API requires pervasive Rc<RefCell<_>> cloning; `Rc::clone(&x)` would be overly verbose.
#![allow(clippy::clone_on_ref_ptr)]
// Physics/rendering formulas are more readable without mul_add transformations.
#![allow(clippy::suboptimal_flops)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::rc::Rc;

use box2d_rs::b2_body::{B2body, B2bodyDef, B2bodyType, BodyPtr};
use box2d_rs::b2_collision::B2manifold;
use box2d_rs::b2_contact::B2contactDynTrait;
use box2d_rs::b2_fixture::{B2filter, B2fixtureDef};
use box2d_rs::b2_joint::B2JointDefEnum;
use box2d_rs::b2_math::B2vec2;
use box2d_rs::b2_world::B2world;
use box2d_rs::b2_world_callbacks::{B2contactImpulse, B2contactListener, B2contactListenerPtr};
use box2d_rs::b2rs_common::UserDataType;
use box2d_rs::joints::b2_revolute_joint::B2revoluteJointDef;
use box2d_rs::shapes::b2_edge_shape::B2edgeShape;
use box2d_rs::shapes::b2_polygon_shape::B2polygonShape;
use rand::RngExt as _;

use crate::env::{Env, EnvMetadata, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::{Error, Result};
use crate::rng::{self, Rng};
use crate::space::{BoundedSpace, Discrete, Space};

const FPS: f32 = 50.0;
const SCALE: f32 = 30.0;

const MAIN_ENGINE_POWER: f32 = 13.0;
const SIDE_ENGINE_POWER: f32 = 0.6;

const INITIAL_RANDOM: f32 = 1000.0;

const LANDER_POLY: [(f32, f32); 6] = [
    (-14.0, 17.0),
    (-17.0, 0.0),
    (-17.0, -10.0),
    (17.0, -10.0),
    (17.0, 0.0),
    (14.0, 17.0),
];

const LEG_AWAY: f32 = 20.0;
const LEG_DOWN: f32 = 18.0;
const LEG_W: f32 = 2.0;
const LEG_H: f32 = 8.0;
const LEG_SPRING_TORQUE: f32 = 40.0;

const SIDE_ENGINE_HEIGHT: f32 = 14.0;
const SIDE_ENGINE_AWAY: f32 = 12.0;
const MAIN_ENGINE_Y_LOCATION: f32 = 4.0;

const VIEWPORT_W: f32 = 600.0;
const VIEWPORT_H: f32 = 400.0;

const CHUNKS: usize = 11;

/// Custom user data attached to `Box2D` bodies.
#[derive(Clone, Debug, Default)]
struct LanderUserData {
    /// Whether this leg body is in contact with the ground.
    ground_contact: bool,
}

/// The `UserDataType` implementation required by `box2d-rs`.
#[derive(Clone, Debug, Default)]
struct LanderData;

impl UserDataType for LanderData {
    type Fixture = ();
    type Body = LanderUserData;
    type Joint = ();
}

/// Tracks ground contact for lander legs and game-over on body contact.
struct ContactDetector {
    /// Leg bodies in the world (set after reset).
    leg_bodies: Vec<BodyPtr<LanderData>>,
    /// The main lander body.
    lander_body: Option<BodyPtr<LanderData>>,
    /// Set to true if the lander body itself touches the ground.
    game_over: Rc<RefCell<bool>>,
}

impl B2contactListener<LanderData> for ContactDetector {
    fn begin_contact(&mut self, contact: &mut dyn B2contactDynTrait<LanderData>) {
        let base = contact.get_base();
        let body_a = base.get_fixture_a().borrow().get_body();
        let body_b = base.get_fixture_b().borrow().get_body();

        // Check if the lander body itself is in contact → game over
        if let Some(ref lander) = self.lander_body
            && (Rc::ptr_eq(&body_a, lander) || Rc::ptr_eq(&body_b, lander))
        {
            *self.game_over.borrow_mut() = true;
        }

        // Check leg ground contact
        for leg in &self.leg_bodies {
            if Rc::ptr_eq(&body_a, leg) || Rc::ptr_eq(&body_b, leg) {
                let mut ud = leg.borrow().get_user_data().unwrap_or_default();
                ud.ground_contact = true;
                leg.borrow_mut().set_user_data(&ud);
            }
        }
    }

    fn end_contact(&mut self, contact: &mut dyn B2contactDynTrait<LanderData>) {
        let base = contact.get_base();
        let body_a = base.get_fixture_a().borrow().get_body();
        let body_b = base.get_fixture_b().borrow().get_body();

        for leg in &self.leg_bodies {
            if Rc::ptr_eq(&body_a, leg) || Rc::ptr_eq(&body_b, leg) {
                let mut ud = leg.borrow().get_user_data().unwrap_or_default();
                ud.ground_contact = false;
                leg.borrow_mut().set_user_data(&ud);
            }
        }
    }

    fn pre_solve(
        &mut self,
        _contact: &mut dyn B2contactDynTrait<LanderData>,
        _old_manifold: &B2manifold,
    ) {
    }

    fn post_solve(
        &mut self,
        _contact: &mut dyn B2contactDynTrait<LanderData>,
        _impulse: &B2contactImpulse,
    ) {
    }
}

/// Configuration for [`LunarLanderEnv`].
#[derive(Debug, Clone, Copy)]
pub struct LunarLanderConfig {
    /// If `true`, use continuous `Box(-1,+1, (2,))` action space.
    /// If `false` (default), use `Discrete(4)`.
    pub continuous: bool,
    /// Gravitational acceleration (must be in `(-12, 0)`). Default: `-10.0`.
    pub gravity: f32,
    /// Enable wind effects. Default: `false`.
    pub enable_wind: bool,
    /// Maximum magnitude of linear wind. Default: `15.0`.
    pub wind_power: f32,
    /// Maximum magnitude of rotational turbulence. Default: `1.5`.
    pub turbulence_power: f32,
    /// Render mode.
    pub render_mode: RenderMode,
}

impl Default for LunarLanderConfig {
    fn default() -> Self {
        Self {
            continuous: false,
            gravity: -10.0,
            enable_wind: false,
            wind_power: 15.0,
            turbulence_power: 1.5,
            render_mode: RenderMode::None,
        }
    }
}

/// Lunar Lander environment with `Box2D` physics.
///
/// The lander starts at the top of the viewport and must land safely on the
/// helipad at coordinates `(0, 0)`. The observation is an 8-dimensional vector
/// and the reward function encourages soft landings on the pad.
///
/// # Action Space
///
/// **Discrete** (`Discrete(4)`):
/// - 0: do nothing
/// - 1: fire left orientation engine
/// - 2: fire main engine
/// - 3: fire right orientation engine
///
/// **Continuous** (`BoundedSpace([-1,-1], [1,1])`):
/// - `[0]`: main engine throttle (`<0` off, `0..1` → 50%-100%)
/// - `[1]`: lateral engines (`|val|<0.5` off, otherwise fire left/right)
///
/// # Observation Space
///
/// `BoundedSpace` of shape `[8]`: `[x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact]`.
///
/// # Rewards
///
/// Reward shaping based on distance/velocity to pad, tilt penalty, leg contact
/// bonus, and fuel cost. +100 for landing, -100 for crashing. Solution ≥ 200.
///
/// Mirrors [Gymnasium `LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/).
pub struct LunarLanderEnv {
    // Spaces
    discrete_action_space: Discrete,
    #[allow(dead_code)] // used by future LunarLanderContinuousEnv
    continuous_action_space: BoundedSpace,
    observation_space: BoundedSpace,

    // Configuration
    continuous: bool,
    gravity: f32,
    enable_wind: bool,
    wind_power: f32,
    turbulence_power: f32,
    render_mode: RenderMode,

    // Box2D world state
    world: Option<box2d_rs::b2_world::B2worldPtr<LanderData>>,
    lander: Option<BodyPtr<LanderData>>,
    legs: Vec<BodyPtr<LanderData>>,
    moon: Option<BodyPtr<LanderData>>,

    // Episode state
    game_over: Rc<RefCell<bool>>,
    prev_shaping: Option<f64>,
    helipad_y: f32,

    // Wind state
    wind_idx: f32,
    torque_idx: f32,

    // Terrain data needed by renderer
    terrain_chunks_x: Vec<f32>,
    terrain_smooth_y: Vec<f32>,
    helipad_x1: f32,
    helipad_x2: f32,

    // Rendering resources (lazily initialized)
    #[cfg(feature = "render")]
    canvas: Option<crate::render::Canvas>,
    #[cfg(feature = "render")]
    window: Option<crate::render::RenderWindow>,

    rng: Rng,
}

impl std::fmt::Debug for LunarLanderEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LunarLanderEnv")
            .field("continuous", &self.continuous)
            .field("gravity", &self.gravity)
            .field("render_mode", &self.render_mode)
            .finish_non_exhaustive()
    }
}

/// Helper: read leg `ground_contact` from user data.
fn leg_ground_contact(leg: &BodyPtr<LanderData>) -> bool {
    leg.borrow()
        .get_user_data()
        .is_some_and(|ud| ud.ground_contact)
}

impl LunarLanderEnv {
    /// Create a new Lunar Lander environment.
    ///
    /// # Errors
    ///
    /// Returns an error if `gravity` is not in `(-12, 0)`.
    pub fn new(config: LunarLanderConfig) -> Result<Self> {
        if config.gravity <= -12.0 || config.gravity >= 0.0 {
            return Err(Error::InvalidSpace {
                reason: format!("gravity must be in (-12, 0), got {}", config.gravity),
            });
        }

        let obs_low = vec![-2.5, -2.5, -10.0, -10.0, -2.0 * PI, -10.0, 0.0, 0.0];
        let obs_high = vec![2.5, 2.5, 10.0, 10.0, 2.0 * PI, 10.0, 1.0, 1.0];

        Ok(Self {
            discrete_action_space: Discrete::new(4),
            continuous_action_space: BoundedSpace::uniform(-1.0, 1.0, 2)?,
            observation_space: BoundedSpace::new(obs_low, obs_high)?,
            continuous: config.continuous,
            gravity: config.gravity,
            enable_wind: config.enable_wind,
            wind_power: config.wind_power,
            turbulence_power: config.turbulence_power,
            render_mode: config.render_mode,
            world: None,
            lander: None,
            legs: Vec::new(),
            moon: None,
            game_over: Rc::new(RefCell::new(false)),
            prev_shaping: None,
            helipad_y: 0.0,
            wind_idx: 0.0,
            torque_idx: 0.0,
            terrain_chunks_x: Vec::new(),
            terrain_smooth_y: Vec::new(),
            helipad_x1: 0.0,
            helipad_x2: 0.0,
            #[cfg(feature = "render")]
            canvas: None,
            #[cfg(feature = "render")]
            window: None,
            rng: rng::create_rng(None),
        })
    }

    /// Destroy all `Box2D` bodies and reset world state.
    fn destroy(&mut self) {
        if let Some(ref world) = self.world {
            let mut w = world.borrow_mut();
            if let Some(moon) = self.moon.take() {
                w.destroy_body(moon);
            }
            for leg in self.legs.drain(..) {
                w.destroy_body(leg);
            }
            if let Some(lander) = self.lander.take() {
                w.destroy_body(lander);
            }
        }
        self.world = None;
    }

    /// Build the terrain, lander body, and legs. Returns initial observation.
    #[allow(clippy::too_many_lines)]
    fn create_world(&mut self) -> Vec<f32> {
        let world = B2world::new(B2vec2::new(0.0, self.gravity));

        let w = VIEWPORT_W / SCALE;
        let h = VIEWPORT_H / SCALE;

        // Generate terrain height map
        let mut height = vec![0.0_f32; CHUNKS + 1];
        for val in &mut height {
            *val = self.rng.random_range(0.0..h / 2.0);
        }
        let mid = CHUNKS / 2;
        self.helipad_y = h / 4.0;
        for i in (mid.saturating_sub(2))..=(mid + 2).min(height.len() - 1) {
            height[i] = self.helipad_y;
        }

        // Smooth heights — Python uses `height[i-1]` which wraps to `height[-1]`
        // (the last element) when i=0, and the coefficient is literally 0.33.
        let smooth_y: Vec<f32> = (0..CHUNKS)
            .map(|i| {
                let prev = if i > 0 { height[i - 1] } else { height[CHUNKS] };
                let next = height[i + 1];
                0.33 * (prev + height[i] + next)
            })
            .collect();

        let chunk_x: Vec<f32> = (0..CHUNKS)
            .map(|i| w / (CHUNKS as f32 - 1.0) * i as f32)
            .collect();

        // Create static moon body with edge fixtures
        let moon_def = B2bodyDef {
            body_type: B2bodyType::B2StaticBody,
            ..B2bodyDef::default()
        };
        let moon = B2world::create_body(world.clone(), &moon_def);

        // Bottom edge
        {
            let mut edge = B2edgeShape::default();
            edge.set_two_sided(B2vec2::new(0.0, 0.0), B2vec2::new(w, 0.0));
            let fd = B2fixtureDef {
                shape: Some(Rc::new(RefCell::new(edge))),
                density: 0.0,
                friction: 0.1,
                ..B2fixtureDef::default()
            };
            B2body::create_fixture(moon.clone(), &fd);
        }

        // Terrain edge segments
        for i in 0..(CHUNKS - 1) {
            let mut edge = B2edgeShape::default();
            edge.set_two_sided(
                B2vec2::new(chunk_x[i], smooth_y[i]),
                B2vec2::new(chunk_x[i + 1], smooth_y[i + 1]),
            );
            let fd = B2fixtureDef {
                shape: Some(Rc::new(RefCell::new(edge))),
                density: 0.0,
                friction: 0.1,
                ..B2fixtureDef::default()
            };
            B2body::create_fixture(moon.clone(), &fd);
        }
        self.moon = Some(moon);

        // Store terrain data for rendering
        self.terrain_chunks_x.clone_from(&chunk_x);
        self.terrain_smooth_y = smooth_y;
        self.helipad_x1 = chunk_x[mid - 1];
        self.helipad_x2 = chunk_x[mid + 1];

        // Create lander body
        let initial_x = VIEWPORT_W / SCALE / 2.0;
        let initial_y = VIEWPORT_H / SCALE;

        let lander_verts: Vec<B2vec2> = LANDER_POLY
            .iter()
            .map(|&(x, y)| B2vec2::new(x / SCALE, y / SCALE))
            .collect();
        let mut lander_shape = B2polygonShape::default();
        lander_shape.set(&lander_verts);

        let lander_def = B2bodyDef {
            body_type: B2bodyType::B2DynamicBody,
            position: B2vec2::new(initial_x, initial_y),
            angle: 0.0,
            ..B2bodyDef::default()
        };
        let lander = B2world::create_body(world.clone(), &lander_def);

        let lander_fd = B2fixtureDef {
            shape: Some(Rc::new(RefCell::new(lander_shape))),
            density: 5.0,
            friction: 0.1,
            restitution: 0.0,
            filter: B2filter {
                category_bits: 0x0010,
                mask_bits: 0x001,
                group_index: 0,
            },
            ..B2fixtureDef::default()
        };
        B2body::create_fixture(lander.clone(), &lander_fd);

        // Apply initial random impulse
        let fx = self.rng.random_range(-INITIAL_RANDOM..INITIAL_RANDOM);
        let fy = self.rng.random_range(-INITIAL_RANDOM..INITIAL_RANDOM);
        lander
            .borrow_mut()
            .apply_force_to_center(B2vec2::new(fx, fy), true);

        // Wind indices
        if self.enable_wind {
            self.wind_idx = self.rng.random_range(-9999_i32..9999) as f32;
            self.torque_idx = self.rng.random_range(-9999_i32..9999) as f32;
        }

        // Create legs with revolute joints
        let mut legs = Vec::with_capacity(2);
        for &side in &[-1.0_f32, 1.0] {
            let mut leg_shape = B2polygonShape::default();
            leg_shape.set_as_box(LEG_W / SCALE, LEG_H / SCALE);

            let leg_def = B2bodyDef {
                body_type: B2bodyType::B2DynamicBody,
                position: B2vec2::new(initial_x - side * LEG_AWAY / SCALE, initial_y),
                angle: side * 0.05,
                user_data: Some(LanderUserData {
                    ground_contact: false,
                }),
                ..B2bodyDef::default()
            };
            let leg = B2world::create_body(world.clone(), &leg_def);

            let leg_fd = B2fixtureDef {
                shape: Some(Rc::new(RefCell::new(leg_shape))),
                density: 1.0,
                restitution: 0.0,
                filter: B2filter {
                    category_bits: 0x0020,
                    mask_bits: 0x001,
                    group_index: 0,
                },
                ..B2fixtureDef::default()
            };
            B2body::create_fixture(leg.clone(), &leg_fd);

            // Revolute joint connecting leg to lander
            let mut rjd = B2revoluteJointDef::default();
            rjd.base.body_a = Some(lander.clone());
            rjd.base.body_b = Some(leg.clone());
            rjd.local_anchor_a = B2vec2::new(0.0, 0.0);
            rjd.local_anchor_b = B2vec2::new(side * LEG_AWAY / SCALE, LEG_DOWN / SCALE);
            rjd.enable_motor = true;
            rjd.enable_limit = true;
            rjd.max_motor_torque = LEG_SPRING_TORQUE;
            rjd.motor_speed = 0.3 * side;

            if side < 0.0 {
                rjd.lower_angle = 0.9 - 0.5;
                rjd.upper_angle = 0.9;
            } else {
                rjd.lower_angle = -0.9;
                rjd.upper_angle = -0.9 + 0.5;
            }

            world
                .borrow_mut()
                .create_joint(&B2JointDefEnum::RevoluteJoint(rjd));
            legs.push(leg);
        }

        // Set up contact detector
        *self.game_over.borrow_mut() = false;
        let detector = ContactDetector {
            leg_bodies: legs.clone(),
            lander_body: Some(lander.clone()),
            game_over: self.game_over.clone(),
        };
        let listener: B2contactListenerPtr<LanderData> = Rc::new(RefCell::new(detector));
        world.borrow_mut().set_contact_listener(listener);

        self.lander = Some(lander);
        self.legs = legs;
        self.world = Some(world);
        self.prev_shaping = None;

        // Perform one no-op step to get initial observation.
        // Must use the actual action path (not Nop) so that dispersion RNG
        // values are consumed, keeping the random stream aligned with Gymnasium.
        let init_action = if self.continuous {
            StepAction::Continuous(0.0, 0.0)
        } else {
            StepAction::Discrete(0)
        };
        self.do_step(&init_action).0
    }

    /// Compute the 8-dimensional state vector from current `Box2D` state.
    fn get_state(&self) -> Vec<f32> {
        let lander = self.lander.as_ref().expect("lander must exist");
        let b = lander.borrow();
        let pos = b.get_position();
        let vel = b.get_linear_velocity();
        let angle = b.get_angle();
        let angular_vel = b.get_angular_velocity();

        let leg0 = leg_ground_contact(&self.legs[0]);
        let leg1 = leg_ground_contact(&self.legs[1]);

        vec![
            (pos.x - VIEWPORT_W / SCALE / 2.0) / (VIEWPORT_W / SCALE / 2.0),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2.0),
            vel.x * (VIEWPORT_W / SCALE / 2.0) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2.0) / FPS,
            angle,
            20.0 * angular_vel / FPS,
            if leg0 { 1.0 } else { 0.0 },
            if leg1 { 1.0 } else { 0.0 },
        ]
    }

    /// Perform one physics step. Returns `(state, m_power, s_power)`.
    fn do_step(&mut self, action: &StepAction) -> (Vec<f32>, f32, f32) {
        let world = self.world.as_ref().expect("world must exist").clone();
        let lander = self.lander.as_ref().expect("lander must exist").clone();

        // Apply wind forces
        if self.enable_wind {
            let on_ground = leg_ground_contact(&self.legs[0]) || leg_ground_contact(&self.legs[1]);
            if !on_ground {
                let wind_mag = (0.02 * self.wind_idx)
                    .sin()
                    .mul_add(1.0, (PI * 0.01 * self.wind_idx).sin())
                    .tanh()
                    * self.wind_power;
                self.wind_idx += 1.0;
                lander
                    .borrow_mut()
                    .apply_force_to_center(B2vec2::new(wind_mag, 0.0), true);

                let torque_mag = (0.02 * self.torque_idx)
                    .sin()
                    .mul_add(1.0, (PI * 0.01 * self.torque_idx).sin())
                    .tanh()
                    * self.turbulence_power;
                self.torque_idx += 1.0;
                lander.borrow_mut().apply_torque(torque_mag, true);
            }
        }

        // Decode action and apply forces
        let (m_power, s_power) = match *action {
            StepAction::Nop => (0.0_f32, 0.0_f32),
            StepAction::Discrete(a) => self.apply_discrete(a, &lander),
            StepAction::Continuous(main_t, lat) => self.apply_continuous(main_t, lat, &lander),
        };

        // Step the physics world
        world.borrow_mut().step(1.0 / FPS, 6 * 30, 2 * 30);

        (self.get_state(), m_power, s_power)
    }

    /// Apply a discrete action (0-3) and return (`main_power`, `side_power`).
    fn apply_discrete(&mut self, action: i64, lander: &BodyPtr<LanderData>) -> (f32, f32) {
        let b = lander.borrow();
        let angle = b.get_angle();
        let pos = b.get_position();
        drop(b);

        let tip = (angle.sin(), angle.cos());
        let side = (-tip.1, tip.0);
        let d0 = self.rng_dispersion();
        let d1 = self.rng_dispersion();

        let mut m_power = 0.0_f32;
        let mut s_power = 0.0_f32;

        // Main engine (action == 2)
        if action == 2 {
            m_power = 1.0;
            let ox = tip.0.mul_add(
                2.0f32.mul_add(d0, MAIN_ENGINE_Y_LOCATION / SCALE),
                side.0 * d1,
            );
            let oy = (-tip.1).mul_add(
                2.0f32.mul_add(d0, MAIN_ENGINE_Y_LOCATION / SCALE),
                -(side.1 * d1),
            );
            let ip = B2vec2::new(pos.x + ox, pos.y + oy);
            lander.borrow_mut().apply_linear_impulse(
                B2vec2::new(
                    -ox * MAIN_ENGINE_POWER * m_power,
                    -oy * MAIN_ENGINE_POWER * m_power,
                ),
                ip,
                true,
            );
        }

        // Side engines (action == 1 or 3)
        if action == 1 || action == 3 {
            let direction = (action - 2) as f32;
            s_power = 1.0;
            let ox = tip.0.mul_add(
                d0,
                side.0 * 3.0f32.mul_add(d1, direction * SIDE_ENGINE_AWAY / SCALE),
            );
            let oy = (-tip.1).mul_add(
                d0,
                -(side.1 * 3.0f32.mul_add(d1, direction * SIDE_ENGINE_AWAY / SCALE)),
            );
            let ip = B2vec2::new(
                pos.x + ox - tip.0 * 17.0 / SCALE,
                pos.y + oy + tip.1 * SIDE_ENGINE_HEIGHT / SCALE,
            );
            lander.borrow_mut().apply_linear_impulse(
                B2vec2::new(
                    -ox * SIDE_ENGINE_POWER * s_power,
                    -oy * SIDE_ENGINE_POWER * s_power,
                ),
                ip,
                true,
            );
        }
        (m_power, s_power)
    }

    /// Apply a continuous action `[main, lateral]` and return (`main_power`, `side_power`).
    fn apply_continuous(
        &mut self,
        main_throttle: f32,
        lateral: f32,
        lander: &BodyPtr<LanderData>,
    ) -> (f32, f32) {
        let b = lander.borrow();
        let angle = b.get_angle();
        let pos = b.get_position();
        drop(b);

        let tip = (angle.sin(), angle.cos());
        let side = (-tip.1, tip.0);
        let d0 = self.rng_dispersion();
        let d1 = self.rng_dispersion();

        let mut m_power = 0.0_f32;
        let mut s_power = 0.0_f32;

        // Main engine
        if main_throttle > 0.0 {
            m_power = (main_throttle.clamp(0.0, 1.0) + 1.0) * 0.5;
            let ox = tip.0.mul_add(
                2.0f32.mul_add(d0, MAIN_ENGINE_Y_LOCATION / SCALE),
                side.0 * d1,
            );
            let oy = (-tip.1).mul_add(
                2.0f32.mul_add(d0, MAIN_ENGINE_Y_LOCATION / SCALE),
                -(side.1 * d1),
            );
            let ip = B2vec2::new(pos.x + ox, pos.y + oy);
            lander.borrow_mut().apply_linear_impulse(
                B2vec2::new(
                    -ox * MAIN_ENGINE_POWER * m_power,
                    -oy * MAIN_ENGINE_POWER * m_power,
                ),
                ip,
                true,
            );
        }

        // Side engines
        if lateral.abs() > 0.5 {
            let direction = lateral.signum();
            s_power = lateral.abs().clamp(0.5, 1.0);
            let ox = tip.0.mul_add(
                d0,
                side.0 * 3.0f32.mul_add(d1, direction * SIDE_ENGINE_AWAY / SCALE),
            );
            let oy = (-tip.1).mul_add(
                d0,
                -(side.1 * 3.0f32.mul_add(d1, direction * SIDE_ENGINE_AWAY / SCALE)),
            );
            let ip = B2vec2::new(
                pos.x + ox - tip.0 * 17.0 / SCALE,
                pos.y + oy + tip.1 * SIDE_ENGINE_HEIGHT / SCALE,
            );
            lander.borrow_mut().apply_linear_impulse(
                B2vec2::new(
                    -ox * SIDE_ENGINE_POWER * s_power,
                    -oy * SIDE_ENGINE_POWER * s_power,
                ),
                ip,
                true,
            );
        }
        (m_power, s_power)
    }

    /// Generate a random dispersion value in `[-1/SCALE, +1/SCALE]`.
    fn rng_dispersion(&mut self) -> f32 {
        self.rng.random_range(-1.0_f32..1.0) / SCALE
    }

    /// Render the scene to an internal canvas, returning the appropriate frame.
    ///
    /// Draws sky polygons (black), terrain, lander body (rotated polygon),
    /// legs, and helipad flags — matching Gymnasium's visual output.
    #[cfg(feature = "render")]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    fn render_pixels(&mut self) -> Result<RenderFrame> {
        use crate::render::{Canvas, RenderWindow};

        if self.lander.is_none() {
            return Err(Error::ResetNeeded { method: "render" });
        }

        let vw = VIEWPORT_W as u32;
        let vh = VIEWPORT_H as u32;

        let canvas = self.canvas.get_or_insert_with(|| Canvas::new(vw, vh));
        // White background (Gymnasium draws white then overlays black sky)
        canvas.clear(tiny_skia::Color::WHITE);

        let h = VIEWPORT_H / SCALE;
        for i in 0..(self.terrain_chunks_x.len().saturating_sub(1)) {
            let x1 = self.terrain_chunks_x[i] * SCALE;
            let y1 = vh as f32 - self.terrain_smooth_y[i] * SCALE;
            let x2 = self.terrain_chunks_x[i + 1] * SCALE;
            let y2 = vh as f32 - self.terrain_smooth_y[i + 1] * SCALE;
            let sky = [
                (x1, y1),
                (x2, y2),
                (x2, vh as f32 - h * SCALE),
                (x1, vh as f32 - h * SCALE),
            ];
            canvas.fill_polygon(&sky, tiny_skia::Color::BLACK);
        }

        // Helper: transform body-local polygon vertices to screen coords
        let draw_body_poly = |canvas: &mut Canvas,
                              body: &BodyPtr<LanderData>,
                              verts: &[(f32, f32)],
                              fill_color: tiny_skia::Color,
                              outline_color: tiny_skia::Color| {
            let b = body.borrow();
            let pos = b.get_position();
            let angle = b.get_angle();
            let cos = angle.cos();
            let sin = angle.sin();

            let screen: Vec<(f32, f32)> = verts
                .iter()
                .map(|&(lx, ly)| {
                    let wx = pos.x + cos * lx - sin * ly;
                    let wy = pos.y + sin * lx + cos * ly;
                    (wx * SCALE, vh as f32 - wy * SCALE)
                })
                .collect();

            canvas.fill_polygon(&screen, fill_color);
            canvas.stroke_polygon(&screen, 1.0, outline_color);
        };

        // Lander body
        let lander = self.lander.as_ref().expect("checked above");
        let lander_verts: Vec<(f32, f32)> = LANDER_POLY
            .iter()
            .map(|&(x, y)| (x / SCALE, y / SCALE))
            .collect();
        let lander_fill = tiny_skia::Color::from_rgba8(230, 230, 230, 255);
        let lander_outline = tiny_skia::Color::from_rgba8(64, 64, 64, 255);
        draw_body_poly(canvas, lander, &lander_verts, lander_fill, lander_outline);

        // Legs
        let leg_w = LEG_W / SCALE;
        let leg_h = LEG_H / SCALE;
        let leg_verts: Vec<(f32, f32)> = vec![
            (-leg_w, -leg_h),
            (leg_w, -leg_h),
            (leg_w, leg_h),
            (-leg_w, leg_h),
        ];
        for (idx, leg) in self.legs.iter().enumerate() {
            let contact = leg_ground_contact(leg);
            let fill = if contact {
                tiny_skia::Color::from_rgba8(0, 200, 0, 255)
            } else {
                tiny_skia::Color::from_rgba8(128, 102, 230, 255)
            };
            let outline = if contact {
                tiny_skia::Color::from_rgba8(0, 140, 0, 255)
            } else {
                tiny_skia::Color::from_rgba8(77, 77, 128, 255)
            };
            // Clone body ptr to satisfy borrow rules in closure
            let leg_clone = leg.clone();
            // Inline the draw since closure borrows canvas mutably
            {
                let b = leg_clone.borrow();
                let pos = b.get_position();
                let angle = b.get_angle();
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                let screen: Vec<(f32, f32)> = leg_verts
                    .iter()
                    .map(|&(lx, ly)| {
                        let wx = pos.x + cos_a * lx - sin_a * ly;
                        let wy = pos.y + sin_a * lx + cos_a * ly;
                        (wx * SCALE, vh as f32 - wy * SCALE)
                    })
                    .collect();

                canvas.fill_polygon(&screen, fill);
                canvas.stroke_polygon(&screen, 1.0, outline);
            }
            let _ = idx; // suppress unused warning
        }

        for &hx in &[self.helipad_x1, self.helipad_x2] {
            let sx = hx * SCALE;
            let flag_y1 = vh as f32 - self.helipad_y * SCALE;
            let flag_y2 = flag_y1 - 50.0;
            // Flagpole
            canvas.stroke_line(sx, flag_y1, sx, flag_y2, 1.0, tiny_skia::Color::WHITE);
            // Flag triangle
            let flag_color = tiny_skia::Color::from_rgba8(204, 204, 0, 255);
            let tri = [
                (sx, flag_y2),
                (sx, flag_y2 + 10.0),
                (sx + 25.0, flag_y2 + 5.0),
            ];
            canvas.fill_polygon(&tri, flag_color);
            canvas.stroke_polygon(&tri, 1.0, flag_color);
        }

        match self.render_mode {
            RenderMode::Human => {
                let window = self.window.get_or_insert_with(|| {
                    RenderWindow::new(
                        "LunarLander \u{2014} gmgn",
                        vw as usize,
                        vh as usize,
                        FPS as usize,
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
                    width: vw,
                    height: vh,
                    data: rgb,
                })
            }
            _ => Ok(RenderFrame::None),
        }
    }
}

/// Internal action representation for the step function.
#[allow(dead_code)] // Continuous variant used by future LunarLanderContinuousEnv
enum StepAction {
    Nop,
    Discrete(i64),
    Continuous(f32, f32),
}

impl Env for LunarLanderEnv {
    type Obs = Vec<f32>;
    type Act = i64;
    type ObsSpace = BoundedSpace;
    type ActSpace = Discrete;

    fn step(&mut self, action: &i64) -> Result<StepResult<Vec<f32>>> {
        if self.world.is_none() {
            return Err(Error::ResetNeeded { method: "step" });
        }

        if self.continuous {
            return Err(Error::InvalidAction {
                reason: "continuous LunarLander requires Vec<f32> actions, \
                         use LunarLanderContinuousEnv instead"
                    .to_owned(),
            });
        }

        if !self.discrete_action_space.contains(action) {
            return Err(Error::InvalidAction {
                reason: format!("expected 0..3, got {action}"),
            });
        }

        let (state, m_power, s_power) = self.do_step(&StepAction::Discrete(*action));

        // Compute reward shaping (use f64 for precision, matching Gymnasium)
        let shaping = 10.0f64.mul_add(
            f64::from(state[7]),
            10.0f64.mul_add(
                f64::from(state[6]),
                100.0f64.mul_add(
                    -f64::from(state[4]).abs(),
                    (-100.0f64).mul_add(
                        f64::from(state[0]).hypot(f64::from(state[1])),
                        -(100.0 * f64::from(state[2]).hypot(f64::from(state[3]))),
                    ),
                ),
            ),
        );

        let mut reward = self.prev_shaping.map_or(0.0, |prev| shaping - prev);
        self.prev_shaping = Some(shaping);

        reward -= f64::from(m_power) * 0.30;
        reward -= f64::from(s_power) * 0.03;

        let game_over = *self.game_over.borrow();
        let lander_awake = self.lander.as_ref().is_some_and(|l| l.borrow().is_awake());

        let terminated = if game_over || state[0].abs() >= 1.0 {
            reward = -100.0;
            true
        } else if !lander_awake {
            reward = 100.0;
            true
        } else {
            false
        };

        Ok(StepResult {
            obs: state,
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

        self.destroy();
        let obs = self.create_world();

        Ok(ResetResult {
            obs,
            info: HashMap::new(),
        })
    }

    fn render(&mut self) -> Result<RenderFrame> {
        match self.render_mode {
            RenderMode::None | RenderMode::Ansi => Ok(RenderFrame::None),
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
        &self.discrete_action_space
    }

    fn render_mode(&self) -> &RenderMode {
        &self.render_mode
    }

    fn metadata(&self) -> EnvMetadata {
        EnvMetadata {
            render_modes: &["human", "rgb_array"],
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            render_fps: Some(FPS as u32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_reset() {
        let mut env =
            LunarLanderEnv::new(LunarLanderConfig::default()).expect("failed to create env");
        let result = env.reset(Some(42)).expect("failed to reset");
        assert_eq!(result.obs.len(), 8, "observation must be 8-dim");
    }

    #[test]
    fn step_discrete_actions() {
        let mut env =
            LunarLanderEnv::new(LunarLanderConfig::default()).expect("failed to create env");
        env.reset(Some(42)).expect("failed to reset");

        for action in 0..4_i64 {
            let result = env.step(&action).expect("step failed");
            assert_eq!(result.obs.len(), 8);
            assert!(result.reward.is_finite(), "reward must be finite");
        }
    }

    #[test]
    fn rejects_step_before_reset() {
        let mut env =
            LunarLanderEnv::new(LunarLanderConfig::default()).expect("failed to create env");
        assert!(env.step(&0).is_err());
    }

    #[test]
    fn rejects_invalid_gravity() {
        assert!(
            LunarLanderEnv::new(LunarLanderConfig {
                gravity: 0.0,
                ..LunarLanderConfig::default()
            })
            .is_err()
        );

        assert!(
            LunarLanderEnv::new(LunarLanderConfig {
                gravity: -12.0,
                ..LunarLanderConfig::default()
            })
            .is_err()
        );
    }

    #[test]
    fn episode_terminates() {
        let mut env =
            LunarLanderEnv::new(LunarLanderConfig::default()).expect("failed to create env");
        env.reset(Some(123)).expect("failed to reset");

        let mut terminated = false;
        for _ in 0..1000 {
            let result = env.step(&0).expect("step failed");
            if result.terminated {
                terminated = true;
                break;
            }
        }
        assert!(
            terminated,
            "episode should terminate within 1000 no-op steps"
        );
    }

    #[test]
    fn seed_determinism() {
        let mut env1 = LunarLanderEnv::new(LunarLanderConfig::default()).expect("create env1");
        let mut env2 = LunarLanderEnv::new(LunarLanderConfig::default()).expect("create env2");

        let r1 = env1.reset(Some(99)).expect("reset env1");
        let r2 = env2.reset(Some(99)).expect("reset env2");
        assert_eq!(r1.obs, r2.obs, "same seed must produce same initial obs");
    }
}
