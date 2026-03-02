//! `BipedalWalker` environment using `Box2D` physics.
//!
//! A simple 4-joint walker robot environment. Two versions:
//! - Normal: slightly uneven terrain
//! - Hardcore: ladders, stumps, pitfalls
//!
//! Continuous action space with 4 motor controls for hip and knee joints.
//!
//! Mirrors [Gymnasium `BipedalWalker-v3`](https://gymnasium.farama.org/environments/box2d/bipedal_walker/).

// box2d-rs API requires pervasive Rc<RefCell<_>> cloning.
#![allow(clippy::clone_on_ref_ptr)]
// Physics formulas are more readable without mul_add transformations.
#![allow(clippy::suboptimal_flops)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::rc::Rc;

use box2d_rs::b2_body::{B2body, B2bodyDef, B2bodyType, BodyPtr};
use box2d_rs::b2_collision::B2manifold;
use box2d_rs::b2_contact::B2contactDynTrait;
use box2d_rs::b2_fixture::{B2filter, B2fixtureDef, FixturePtr};
use box2d_rs::b2_joint::{B2JointDefEnum, B2jointPtr, JointAsDerived, JointAsDerivedMut};
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
use crate::space::BoundedSpace;

const FPS: f32 = 50.0;
const SCALE: f32 = 30.0;

const MOTORS_TORQUE: f32 = 80.0;
const SPEED_HIP: f32 = 4.0;
const SPEED_KNEE: f32 = 6.0;
const LIDAR_RANGE: f32 = 160.0 / SCALE;

const INITIAL_RANDOM: f32 = 5.0;

const HULL_POLY: [(f32, f32); 5] = [
    (-30.0, 9.0),
    (6.0, 9.0),
    (34.0, 1.0),
    (34.0, -8.0),
    (-30.0, -8.0),
];

const LEG_DOWN: f32 = -8.0 / SCALE;
const LEG_W: f32 = 8.0 / SCALE;
const LEG_H: f32 = 34.0 / SCALE;

const VIEWPORT_W: f32 = 600.0;
const VIEWPORT_H: f32 = 400.0;

const TERRAIN_STEP: f32 = 14.0 / SCALE;
const TERRAIN_LENGTH: usize = 200;
const TERRAIN_HEIGHT: f32 = VIEWPORT_H / SCALE / 4.0;
const TERRAIN_GRASS: usize = 10;
const TERRAIN_STARTPAD: usize = 20;
const FRICTION: f32 = 2.5;

// Collision categories
const CAT_GROUND: u16 = 0x0001;
const CAT_WALKER: u16 = 0x0020;
const MASK_GROUND_ONLY: u16 = 0x001;

// Terrain generation states (hardcore mode)
const GRASS: u8 = 0;
const STUMP: u8 = 1;
const STAIRS: u8 = 2;
const PIT: u8 = 3;
const NUM_OBSTACLE_STATES: u8 = 4; // exclusive upper bound for random

#[derive(Default, Clone, Debug)]
struct WalkerUserData {
    ground_contact: bool,
}

#[derive(Clone, Default, Debug)]
struct WalkerData;

impl UserDataType for WalkerData {
    type Fixture = ();
    type Body = WalkerUserData;
    type Joint = ();
}

struct ContactDetector {
    hull_body: Option<BodyPtr<WalkerData>>,
    /// Lower leg bodies (indices 1 and 3 in Python's `self.legs`).
    lower_leg_bodies: Vec<BodyPtr<WalkerData>>,
    game_over: Rc<RefCell<bool>>,
}

impl B2contactListener<WalkerData> for ContactDetector {
    fn begin_contact(&mut self, contact: &mut dyn B2contactDynTrait<WalkerData>) {
        let base = contact.get_base();
        let body_a = base.get_fixture_a().borrow().get_body();
        let body_b = base.get_fixture_b().borrow().get_body();

        // Hull contact with ground → game over
        if let Some(hull) = &self.hull_body
            && (Rc::ptr_eq(&body_a, hull) || Rc::ptr_eq(&body_b, hull))
        {
            *self.game_over.borrow_mut() = true;
        }

        // Lower legs contact → set ground_contact flag
        for leg in &self.lower_leg_bodies {
            if Rc::ptr_eq(&body_a, leg) || Rc::ptr_eq(&body_b, leg) {
                let mut ud = leg.borrow().get_user_data().unwrap_or_default();
                ud.ground_contact = true;
                leg.borrow_mut().set_user_data(&ud);
            }
        }
    }

    fn end_contact(&mut self, contact: &mut dyn B2contactDynTrait<WalkerData>) {
        let base = contact.get_base();
        let body_a = base.get_fixture_a().borrow().get_body();
        let body_b = base.get_fixture_b().borrow().get_body();

        for leg in &self.lower_leg_bodies {
            if Rc::ptr_eq(&body_a, leg) || Rc::ptr_eq(&body_b, leg) {
                let mut ud = leg.borrow().get_user_data().unwrap_or_default();
                ud.ground_contact = false;
                leg.borrow_mut().set_user_data(&ud);
            }
        }
    }

    fn pre_solve(
        &mut self,
        _contact: &mut dyn B2contactDynTrait<WalkerData>,
        _old_manifold: &B2manifold,
    ) {
    }

    fn post_solve(
        &mut self,
        _contact: &mut dyn B2contactDynTrait<WalkerData>,
        _impulse: &B2contactImpulse,
    ) {
    }
}

/// Configuration for [`BipedalWalkerEnv`].
#[derive(Debug, Clone, Copy)]
pub struct BipedalWalkerConfig {
    /// Enable hardcore mode with obstacles (stumps, stairs, pits).
    /// Default: `false`.
    pub hardcore: bool,
    /// Render mode. Default: [`RenderMode::None`].
    pub render_mode: RenderMode,
}

impl Default for BipedalWalkerConfig {
    fn default() -> Self {
        Self {
            hardcore: false,
            render_mode: RenderMode::None,
        }
    }
}

/// BipedalWalker-v3: a simple 4-joint walking robot.
///
/// # Observation Space
///
/// 24-dimensional continuous: hull angle/velocity, joint angles/speeds,
/// leg ground contacts, and 10 LIDAR rangefinder measurements.
///
/// # Action Space
///
/// 4-dimensional continuous `Box(-1, +1, (4,))`: motor speed values for
/// hip and knee joints on both legs.
///
/// # Rewards
///
/// +300 for reaching the far end. Forward movement is rewarded, −100 for
/// falling. Motor torque costs a small amount. Solution ≥ 300.
///
/// Mirrors [Gymnasium `BipedalWalker-v3`](https://gymnasium.farama.org/environments/box2d/bipedal_walker/).
pub struct BipedalWalkerEnv {
    // Spaces
    action_space: BoundedSpace,
    observation_space: BoundedSpace,

    // Configuration
    hardcore: bool,
    render_mode: RenderMode,

    // Physics state
    world: Option<Rc<RefCell<B2world<WalkerData>>>>,
    hull: Option<BodyPtr<WalkerData>>,
    /// All 4 leg bodies: `[upper_left, lower_left, upper_right, lower_right]`.
    legs: Vec<BodyPtr<WalkerData>>,
    /// All 4 revolute joints: `[hip_left, knee_left, hip_right, knee_right]`.
    joints: Vec<B2jointPtr<WalkerData>>,
    terrain_bodies: Vec<BodyPtr<WalkerData>>,

    // Episode state
    game_over: Rc<RefCell<bool>>,
    prev_shaping: Option<f64>,
    scroll: f32,

    // Terrain rendering data: (4 vertices, fill color RGB)
    #[allow(clippy::type_complexity)]
    terrain_poly: Vec<([(f32, f32); 4], (u8, u8, u8))>,
    // Cloud rendering data: (polygon vertices, x_min, x_max)
    cloud_poly: Vec<(Vec<(f32, f32)>, f32, f32)>,
    // LIDAR animation counter (used for animated ray display)
    #[cfg(feature = "render")]
    #[allow(dead_code)]
    lidar_render: usize,

    // Rendering resources (lazily initialized)
    #[cfg(feature = "render")]
    canvas: Option<crate::render::Canvas>,
    #[cfg(feature = "render")]
    window: Option<crate::render::RenderWindow>,

    // RNG
    rng: Rng,
}

impl std::fmt::Debug for BipedalWalkerEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BipedalWalkerEnv")
            .field("hardcore", &self.hardcore)
            .field("render_mode", &self.render_mode)
            .finish_non_exhaustive()
    }
}

impl BipedalWalkerEnv {
    /// Create a new `BipedalWalker` environment.
    ///
    /// # Errors
    ///
    /// Returns an error if the observation/action space cannot be constructed.
    pub fn new(config: BipedalWalkerConfig) -> Result<Self> {
        // Observation space: 24-dim (matching Gymnasium exactly)
        // [hull_angle, hull_angvel, vel_x, vel_y,
        //  hip0_angle, hip0_speed, knee0_angle, knee0_speed, leg0_contact,
        //  hip1_angle, hip1_speed, knee1_angle, knee1_speed, leg1_contact,
        //  lidar_0..lidar_9]
        let obs_low: Vec<f32> = [
            -PI, -5.0, -5.0, -5.0, -PI, -5.0, -PI, -5.0, 0.0, -PI, -5.0, -PI, -5.0, 0.0,
        ]
        .iter()
        .chain([-1.0_f32; 10].iter())
        .copied()
        .collect();

        let obs_high: Vec<f32> = [
            PI, 5.0, 5.0, 5.0, PI, 5.0, PI, 5.0, 5.0, PI, 5.0, PI, 5.0, 5.0,
        ]
        .iter()
        .chain([1.0_f32; 10].iter())
        .copied()
        .collect();

        let observation_space = BoundedSpace::new(obs_low, obs_high)?;
        let action_space = BoundedSpace::new(vec![-1.0; 4], vec![1.0; 4])?;

        Ok(Self {
            action_space,
            observation_space,
            hardcore: config.hardcore,
            render_mode: config.render_mode,
            world: None,
            hull: None,
            legs: Vec::new(),
            joints: Vec::new(),
            terrain_bodies: Vec::new(),
            game_over: Rc::new(RefCell::new(false)),
            prev_shaping: None,
            scroll: 0.0,
            terrain_poly: Vec::new(),
            cloud_poly: Vec::new(),
            #[cfg(feature = "render")]
            lidar_render: 0,
            #[cfg(feature = "render")]
            canvas: None,
            #[cfg(feature = "render")]
            window: None,
            rng: rng::create_rng(None),
        })
    }

    /// Create the physics world, terrain, hull, and legs. Returns initial obs.
    fn create_world(&mut self) -> Vec<f32> {
        let world = B2world::new(B2vec2::new(0.0, -10.0));

        // Generate terrain and clouds
        self.generate_terrain(&world);
        self.generate_clouds();

        // Create hull
        let init_x = TERRAIN_STEP * TERRAIN_STARTPAD as f32 / 2.0;
        let init_y = TERRAIN_HEIGHT + 2.0 * LEG_H;

        let hull_verts: Vec<B2vec2> = HULL_POLY
            .iter()
            .map(|&(x, y)| B2vec2::new(x / SCALE, y / SCALE))
            .collect();
        let mut hull_shape = B2polygonShape::default();
        hull_shape.set(&hull_verts);

        let hull_def = B2bodyDef {
            body_type: B2bodyType::B2DynamicBody,
            position: B2vec2::new(init_x, init_y),
            ..B2bodyDef::default()
        };
        let hull = B2world::create_body(world.clone(), &hull_def);

        let hull_fd = B2fixtureDef {
            shape: Some(Rc::new(RefCell::new(hull_shape))),
            density: 5.0,
            friction: 0.1,
            restitution: 0.0,
            filter: B2filter {
                category_bits: CAT_WALKER,
                mask_bits: MASK_GROUND_ONLY,
                group_index: 0,
            },
            ..B2fixtureDef::default()
        };
        B2body::create_fixture(hull.clone(), &hull_fd);

        // Apply initial random horizontal force
        let fx = self.rng.random_range(-INITIAL_RANDOM..INITIAL_RANDOM);
        hull.borrow_mut()
            .apply_force_to_center(B2vec2::new(fx, 0.0), true);

        // Create legs and joints
        let mut legs = Vec::with_capacity(4);
        let mut joints = Vec::with_capacity(4);

        for &side in &[-1.0_f32, 1.0] {
            // Upper leg
            let mut upper_shape = B2polygonShape::default();
            upper_shape.set_as_box(LEG_W / 2.0, LEG_H / 2.0);

            let upper_def = B2bodyDef {
                body_type: B2bodyType::B2DynamicBody,
                position: B2vec2::new(init_x, init_y - LEG_H / 2.0 - LEG_DOWN),
                angle: side * 0.05,
                ..B2bodyDef::default()
            };
            let upper = B2world::create_body(world.clone(), &upper_def);

            let leg_fd = B2fixtureDef {
                shape: Some(Rc::new(RefCell::new(upper_shape))),
                density: 1.0,
                restitution: 0.0,
                filter: B2filter {
                    category_bits: CAT_WALKER,
                    mask_bits: MASK_GROUND_ONLY,
                    group_index: 0,
                },
                ..B2fixtureDef::default()
            };
            B2body::create_fixture(upper.clone(), &leg_fd);

            // Hip joint (hull ↔ upper leg)
            let mut hip_jd = B2revoluteJointDef::default();
            hip_jd.base.body_a = Some(hull.clone());
            hip_jd.base.body_b = Some(upper.clone());
            hip_jd.local_anchor_a = B2vec2::new(0.0, LEG_DOWN);
            hip_jd.local_anchor_b = B2vec2::new(0.0, LEG_H / 2.0);
            hip_jd.enable_motor = true;
            hip_jd.enable_limit = true;
            hip_jd.max_motor_torque = MOTORS_TORQUE;
            hip_jd.motor_speed = side;
            hip_jd.lower_angle = -0.8;
            hip_jd.upper_angle = 1.1;

            let hip_joint = world
                .borrow_mut()
                .create_joint(&B2JointDefEnum::RevoluteJoint(hip_jd));

            legs.push(upper.clone());
            joints.push(hip_joint);

            // Lower leg
            let mut lower_shape = B2polygonShape::default();
            lower_shape.set_as_box(0.8 * LEG_W / 2.0, LEG_H / 2.0);

            let lower_def = B2bodyDef {
                body_type: B2bodyType::B2DynamicBody,
                position: B2vec2::new(init_x, init_y - LEG_H * 3.0 / 2.0 - LEG_DOWN),
                angle: side * 0.05,
                user_data: Some(WalkerUserData {
                    ground_contact: false,
                }),
                ..B2bodyDef::default()
            };
            let lower = B2world::create_body(world.clone(), &lower_def);

            let lower_fd = B2fixtureDef {
                shape: Some(Rc::new(RefCell::new(lower_shape))),
                density: 1.0,
                restitution: 0.0,
                filter: B2filter {
                    category_bits: CAT_WALKER,
                    mask_bits: MASK_GROUND_ONLY,
                    group_index: 0,
                },
                ..B2fixtureDef::default()
            };
            B2body::create_fixture(lower.clone(), &lower_fd);

            // Knee joint (upper leg ↔ lower leg)
            let mut knee_jd = B2revoluteJointDef::default();
            knee_jd.base.body_a = Some(upper);
            knee_jd.base.body_b = Some(lower.clone());
            knee_jd.local_anchor_a = B2vec2::new(0.0, -LEG_H / 2.0);
            knee_jd.local_anchor_b = B2vec2::new(0.0, LEG_H / 2.0);
            knee_jd.enable_motor = true;
            knee_jd.enable_limit = true;
            knee_jd.max_motor_torque = MOTORS_TORQUE;
            knee_jd.motor_speed = 1.0;
            knee_jd.lower_angle = -1.6;
            knee_jd.upper_angle = -0.1;

            let knee_joint = world
                .borrow_mut()
                .create_joint(&B2JointDefEnum::RevoluteJoint(knee_jd));

            legs.push(lower);
            joints.push(knee_joint);
        }

        // Set up contact detector
        *self.game_over.borrow_mut() = false;
        let lower_legs = vec![legs[1].clone(), legs[3].clone()];
        let detector = ContactDetector {
            hull_body: Some(hull.clone()),
            lower_leg_bodies: lower_legs,
            game_over: self.game_over.clone(),
        };
        let listener: B2contactListenerPtr<WalkerData> = Rc::new(RefCell::new(detector));
        world.borrow_mut().set_contact_listener(listener);

        self.hull = Some(hull);
        self.legs = legs;
        self.joints = joints;
        self.world = Some(world);
        self.prev_shaping = None;
        self.scroll = 0.0;

        // Perform one no-op step to get initial observation (matching Gymnasium)
        let zero_action = vec![0.0_f32; 4];
        self.do_step(&zero_action).0
    }

    /// Generate terrain height profile and create `Box2D` bodies.
    fn generate_terrain(&mut self, world: &Rc<RefCell<B2world<WalkerData>>>) {
        self.terrain_poly.clear();

        let mut state = GRASS;
        let mut velocity = 0.0_f32;
        let mut y = TERRAIN_HEIGHT;
        let mut counter: usize = TERRAIN_STARTPAD;
        let mut oneshot = false;

        let mut terrain_x = Vec::with_capacity(TERRAIN_LENGTH);
        let mut terrain_y = Vec::with_capacity(TERRAIN_LENGTH);

        // Hardcore obstacle state variables
        let mut stair_steps: i32 = 0;
        let mut stair_width: i32 = 0;
        let mut stair_height: i32 = 0;
        let mut original_y: f32 = 0.0;

        for i in 0..TERRAIN_LENGTH {
            let x = i as f32 * TERRAIN_STEP;
            terrain_x.push(x);

            if state == GRASS && !oneshot {
                velocity = 0.01f32.mul_add((TERRAIN_HEIGHT - y).signum(), 0.8 * velocity);
                if i > TERRAIN_STARTPAD {
                    velocity += self.rng.random_range(-1.0_f32..1.0) / SCALE;
                }
                y += velocity;
            } else if state == PIT && oneshot {
                counter = self.rng.random_range(3..5_usize);
                let poly = [
                    B2vec2::new(x, y),
                    B2vec2::new(x + TERRAIN_STEP, y),
                    B2vec2::new(x + TERRAIN_STEP, y - 4.0 * TERRAIN_STEP),
                    B2vec2::new(x, y - 4.0 * TERRAIN_STEP),
                ];
                self.create_terrain_polygon(world, &poly);

                let offset = TERRAIN_STEP * counter as f32;
                let poly2 = [
                    B2vec2::new(poly[0].x + offset, poly[0].y),
                    B2vec2::new(poly[1].x + offset, poly[1].y),
                    B2vec2::new(poly[2].x + offset, poly[2].y),
                    B2vec2::new(poly[3].x + offset, poly[3].y),
                ];
                self.create_terrain_polygon(world, &poly2);
                counter += 2;
                original_y = y;
            } else if state == PIT && !oneshot {
                y = original_y;
                if counter > 1 {
                    y -= 4.0 * TERRAIN_STEP;
                }
            } else if state == STUMP && oneshot {
                counter = self.rng.random_range(1..3_usize);
                let c = counter as f32;
                let poly = [
                    B2vec2::new(x, y),
                    B2vec2::new(x + c * TERRAIN_STEP, y),
                    B2vec2::new(x + c * TERRAIN_STEP, y + c * TERRAIN_STEP),
                    B2vec2::new(x, y + c * TERRAIN_STEP),
                ];
                self.create_terrain_polygon(world, &poly);
            } else if state == STAIRS && oneshot {
                stair_height = if self.rng.random_range(0.0_f32..1.0) > 0.5 {
                    1
                } else {
                    -1
                };
                stair_width = self.rng.random_range(4..5_i32);
                stair_steps = self.rng.random_range(3..5_i32);
                original_y = y;
                for s in 0..stair_steps {
                    let sx = x + (s * stair_width) as f32 * TERRAIN_STEP;
                    let sx2 = x + ((1 + s) * stair_width) as f32 * TERRAIN_STEP;
                    let sy = y + (s * stair_height) as f32 * TERRAIN_STEP;
                    let sy_low = y + (-1 + s * stair_height) as f32 * TERRAIN_STEP;
                    let poly = [
                        B2vec2::new(sx, sy),
                        B2vec2::new(sx2, sy),
                        B2vec2::new(sx2, sy_low),
                        B2vec2::new(sx, sy_low),
                    ];
                    self.create_terrain_polygon(world, &poly);
                }
                #[allow(clippy::cast_sign_loss)]
                {
                    counter = (stair_steps * stair_width) as usize;
                }
            } else if state == STAIRS && !oneshot {
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                let counter_i32 = counter as i32;
                let s = stair_steps * stair_width - counter_i32 - stair_height;
                let n = s as f32 / stair_width as f32;
                y = n.mul_add(stair_height as f32 * TERRAIN_STEP, original_y);
            }
            // STUMP && !oneshot → y unchanged (terrain stays at same height)

            oneshot = false;
            terrain_y.push(y);
            counter -= 1;
            if counter == 0 {
                counter = self.rng.random_range(TERRAIN_GRASS / 2..TERRAIN_GRASS);
                if state == GRASS && self.hardcore {
                    state = self.rng.random_range(1..NUM_OBSTACLE_STATES);
                } else {
                    state = GRASS;
                }
                oneshot = true;
            }
        }

        // Create terrain edge fixtures and save polygon data for rendering
        // Note: obstacle polygons were already pushed by create_terrain_polygon above;
        // do NOT clear terrain_poly here — it was cleared in generate_terrain's caller.
        for i in 0..(TERRAIN_LENGTH - 1) {
            let mut edge = B2edgeShape::default();
            edge.set_two_sided(
                B2vec2::new(terrain_x[i], terrain_y[i]),
                B2vec2::new(terrain_x[i + 1], terrain_y[i + 1]),
            );

            // Store filled polygon for rendering (terrain surface down to y=0)
            let color: (u8, u8, u8) = if i % 2 == 0 {
                (76, 255, 76)
            } else {
                (76, 204, 76)
            };
            let fill_color = (102, 153, 76);
            let poly = [
                (terrain_x[i], terrain_y[i]),
                (terrain_x[i + 1], terrain_y[i + 1]),
                (terrain_x[i + 1], 0.0),
                (terrain_x[i], 0.0),
            ];
            let _ = color; // edge color stored on body in Python; we use fill_color
            self.terrain_poly.push((poly, fill_color));
            let body_def = B2bodyDef {
                body_type: B2bodyType::B2StaticBody,
                ..B2bodyDef::default()
            };
            let body = B2world::create_body(world.clone(), &body_def);
            let fd = B2fixtureDef {
                shape: Some(Rc::new(RefCell::new(edge))),
                friction: FRICTION,
                filter: B2filter {
                    category_bits: CAT_GROUND,
                    mask_bits: 0xFFFF,
                    group_index: 0,
                },
                ..B2fixtureDef::default()
            };
            B2body::create_fixture(body.clone(), &fd);
            self.terrain_bodies.push(body);
        }
    }

    /// Create a static polygon body for terrain obstacles.
    fn create_terrain_polygon(
        &mut self,
        world: &Rc<RefCell<B2world<WalkerData>>>,
        vertices: &[B2vec2],
    ) {
        let mut shape = B2polygonShape::default();
        shape.set(vertices);
        let body_def = B2bodyDef {
            body_type: B2bodyType::B2StaticBody,
            ..B2bodyDef::default()
        };
        let body = B2world::create_body(world.clone(), &body_def);
        let fd = B2fixtureDef {
            shape: Some(Rc::new(RefCell::new(shape))),
            friction: FRICTION,
            filter: B2filter {
                category_bits: CAT_GROUND,
                mask_bits: 0xFFFF,
                group_index: 0,
            },
            ..B2fixtureDef::default()
        };
        B2body::create_fixture(body.clone(), &fd);
        self.terrain_bodies.push(body);

        // Save obstacle polygon data for rendering (white fill, matching Gymnasium)
        if vertices.len() == 4 {
            let poly = [
                (vertices[0].x, vertices[0].y),
                (vertices[1].x, vertices[1].y),
                (vertices[2].x, vertices[2].y),
                (vertices[3].x, vertices[3].y),
            ];
            self.terrain_poly.push((poly, (255, 255, 255)));
        }
    }

    /// Generate random cloud polygons for rendering, matching Gymnasium's `_generate_clouds`.
    fn generate_clouds(&mut self) {
        self.cloud_poly.clear();
        let num_clouds = TERRAIN_LENGTH / 20;
        for _ in 0..num_clouds {
            let x = self.rng.random_range(0.0..(TERRAIN_LENGTH as f32)) * TERRAIN_STEP;
            let y = VIEWPORT_H / SCALE * 3.0 / 4.0;
            #[allow(clippy::approx_constant)] // Gymnasium intentionally uses 3.14, not math.pi
            let poly: Vec<(f32, f32)> = (0..5)
                .map(|a| {
                    let angle = 3.14 * 2.0 * a as f32 / 5.0;
                    let px = x
                        + 15.0 * TERRAIN_STEP * angle.sin()
                        + self.rng.random_range(0.0..5.0 * TERRAIN_STEP);
                    let py = y
                        + 5.0 * TERRAIN_STEP * angle.cos()
                        + self.rng.random_range(0.0..5.0 * TERRAIN_STEP);
                    (px, py)
                })
                .collect();
            let x1 = poly.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
            let x2 = poly.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
            self.cloud_poly.push((poly, x1, x2));
        }
    }

    /// Render the scene to an internal canvas, returning the appropriate frame.
    ///
    /// Draws sky, clouds (with parallax), terrain polygons, hull, legs, joints,
    /// and a start flag — matching Gymnasium's visual output.
    #[cfg(feature = "render")]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    fn render_pixels(&mut self) -> Result<RenderFrame> {
        use crate::render::{Canvas, RenderWindow};

        if self.hull.is_none() {
            return Err(Error::ResetNeeded { method: "render" });
        }

        let vw = VIEWPORT_W as u32;
        let vh = VIEWPORT_H as u32;
        let vw_f = VIEWPORT_W;
        let vh_f = VIEWPORT_H;

        let canvas = self.canvas.get_or_insert_with(|| Canvas::new(vw, vh));

        // Sky background
        let sky_color = tiny_skia::Color::from_rgba8(215, 215, 255, 255);
        canvas.clear(sky_color);

        // Clouds (parallax at half scroll speed)
        let cloud_color = tiny_skia::Color::WHITE;
        for (poly, x1, x2) in &self.cloud_poly {
            if *x2 < self.scroll / 2.0 {
                continue;
            }
            if *x1 > self.scroll / 2.0 + vw_f / SCALE {
                continue;
            }
            let screen_poly: Vec<(f32, f32)> = poly
                .iter()
                .map(|&(px, py)| {
                    let sx = (px - self.scroll / 2.0) * SCALE;
                    let sy = vh_f - py * SCALE;
                    (sx, sy)
                })
                .collect();
            canvas.fill_polygon(&screen_poly, cloud_color);
        }

        // Terrain polygons
        for (poly, color) in &self.terrain_poly {
            // Visibility culling
            if poly[1].0 < self.scroll {
                continue;
            }
            if poly[0].0 > self.scroll + vw_f / SCALE {
                continue;
            }
            let screen_poly: Vec<(f32, f32)> = poly
                .iter()
                .map(|&(px, py)| {
                    let sx = (px - self.scroll) * SCALE;
                    let sy = vh_f - py * SCALE;
                    (sx, sy)
                })
                .collect();
            let fill = tiny_skia::Color::from_rgba8(color.0, color.1, color.2, 255);
            canvas.fill_polygon(&screen_poly, fill);
        }

        // Draw bodies: hull + legs
        // Body colors matching Gymnasium (hull purple, legs vary by side)
        let body_colors: [(tiny_skia::Color, tiny_skia::Color); 5] = [
            // hull
            (
                tiny_skia::Color::from_rgba8(127, 51, 229, 255),
                tiny_skia::Color::from_rgba8(76, 76, 127, 255),
            ),
            // upper left leg
            (
                tiny_skia::Color::from_rgba8(178, 101, 152, 255),
                tiny_skia::Color::from_rgba8(127, 76, 101, 255),
            ),
            // lower left leg
            (
                tiny_skia::Color::from_rgba8(178, 101, 152, 255),
                tiny_skia::Color::from_rgba8(127, 76, 101, 255),
            ),
            // upper right leg
            (
                tiny_skia::Color::from_rgba8(128, 51, 102, 255),
                tiny_skia::Color::from_rgba8(77, 26, 51, 255),
            ),
            // lower right leg
            (
                tiny_skia::Color::from_rgba8(128, 51, 102, 255),
                tiny_skia::Color::from_rgba8(77, 26, 51, 255),
            ),
        ];

        // Hull vertices (in body-local coords, scaled)
        let hull_local: Vec<(f32, f32)> = HULL_POLY
            .iter()
            .map(|&(x, y)| (x / SCALE, y / SCALE))
            .collect();

        // Leg box vertices (body-local)
        let leg_half_w = LEG_W / 2.0;
        let leg_half_h = LEG_H / 2.0;
        let leg_local: [(f32, f32); 4] = [
            (-leg_half_w, -leg_half_h),
            (leg_half_w, -leg_half_h),
            (leg_half_w, leg_half_h),
            (-leg_half_w, leg_half_h),
        ];

        // Draw order: terrain bodies already drawn, now legs then hull (hull on top)
        // Gymnasium draws in drawlist order: terrain + legs + [hull]
        // We draw legs first, then hull on top
        let hull = self.hull.as_ref().expect("checked above");

        // Draw legs
        for (i, leg) in self.legs.iter().enumerate() {
            let (fill, outline) = body_colors[i + 1];
            let b = leg.borrow();
            let pos = b.get_position();
            let angle = b.get_angle();
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let screen: Vec<(f32, f32)> = leg_local
                .iter()
                .map(|&(lx, ly)| {
                    let wx = pos.x + cos_a * lx - sin_a * ly;
                    let wy = pos.y + sin_a * lx + cos_a * ly;
                    ((wx - self.scroll) * SCALE, vh_f - wy * SCALE)
                })
                .collect();

            canvas.fill_polygon(&screen, fill);
            canvas.stroke_polygon(&screen, 1.0, outline);
        }

        // Draw hull
        {
            let (fill, outline) = body_colors[0];
            let b = hull.borrow();
            let pos = b.get_position();
            let angle = b.get_angle();
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let screen: Vec<(f32, f32)> = hull_local
                .iter()
                .map(|&(lx, ly)| {
                    let wx = pos.x + cos_a * lx - sin_a * ly;
                    let wy = pos.y + sin_a * lx + cos_a * ly;
                    ((wx - self.scroll) * SCALE, vh_f - wy * SCALE)
                })
                .collect();

            canvas.fill_polygon(&screen, fill);
            canvas.stroke_polygon(&screen, 1.0, outline);
        }

        // Start flag
        {
            let flag_x = (TERRAIN_STEP * 3.0 - self.scroll) * SCALE;
            let flag_y1 = vh_f - TERRAIN_HEIGHT * SCALE;
            let flag_y2 = flag_y1 - 50.0;
            canvas.stroke_line(
                flag_x,
                flag_y1,
                flag_x,
                flag_y2,
                1.0,
                tiny_skia::Color::BLACK,
            );
            let flag_color = tiny_skia::Color::from_rgba8(230, 51, 0, 255);
            let tri = [
                (flag_x, flag_y2),
                (flag_x, flag_y2 + 10.0),
                (flag_x + 25.0, flag_y2 + 5.0),
            ];
            canvas.fill_polygon(&tri, flag_color);
            canvas.stroke_polygon(&tri, 1.0, tiny_skia::Color::BLACK);
        }

        // Output
        match self.render_mode {
            RenderMode::Human => {
                let window = self.window.get_or_insert_with(|| {
                    RenderWindow::new(
                        "BipedalWalker \u{2014} gmgn",
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

    /// Compute the 24-dimensional state vector from current physics state.
    fn get_state(&self) -> Vec<f32> {
        let hull = self.hull.as_ref().expect("hull must exist");
        let world = self.world.as_ref().expect("world must exist");

        let hb = hull.borrow();
        let _pos = hb.get_position();
        let vel = hb.get_linear_velocity();
        let angle = hb.get_angle();
        let angular_vel = hb.get_angular_velocity();
        drop(hb);

        let mut state = Vec::with_capacity(24);

        // Hull state (4 values)
        state.push(angle);
        state.push(2.0 * angular_vel / FPS);
        state.push(0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS);
        state.push(0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS);

        // Joint states + leg contacts (5 values per leg pair × 2 = 10)
        for pair in 0..2 {
            let hip_idx = pair * 2;
            let knee_idx = pair * 2 + 1;
            let lower_leg_idx = pair * 2 + 1; // legs[1] and legs[3]

            let hip_joint = self.joints[hip_idx].borrow();
            if let JointAsDerived::ERevoluteJoint(hip) = hip_joint.as_derived() {
                state.push(hip.get_joint_angle());
                state.push(hip.get_joint_speed() / SPEED_HIP);
            }
            drop(hip_joint);

            let knee_joint = self.joints[knee_idx].borrow();
            if let JointAsDerived::ERevoluteJoint(knee) = knee_joint.as_derived() {
                state.push(knee.get_joint_angle() + 1.0);
                state.push(knee.get_joint_speed() / SPEED_KNEE);
            }
            drop(knee_joint);

            let contact = self.legs[lower_leg_idx]
                .borrow()
                .get_user_data()
                .is_some_and(|ud| ud.ground_contact);
            state.push(if contact { 1.0 } else { 0.0 });
        }

        // LIDAR (10 values)
        let hb = hull.borrow();
        let hull_pos = hb.get_position();
        drop(hb);

        for i in 0..10 {
            let ray_angle = 1.5 * i as f32 / 10.0;
            let p1 = hull_pos;
            let p2 = B2vec2::new(
                ray_angle.sin().mul_add(LIDAR_RANGE, hull_pos.x),
                ray_angle.cos().mul_add(-LIDAR_RANGE, hull_pos.y),
            );
            let mut fraction = 1.0_f32;
            world.borrow().ray_cast(
                |fixture: FixturePtr<WalkerData>,
                 _point: B2vec2,
                 _normal: B2vec2,
                 frac: f32|
                 -> f32 {
                    // Only detect ground fixtures (categoryBits & 1 != 0)
                    if (fixture.borrow().get_filter_data().category_bits & 1) == 0 {
                        return -1.0;
                    }
                    fraction = frac;
                    frac
                },
                p1,
                p2,
            );
            state.push(fraction);
        }

        debug_assert_eq!(state.len(), 24);
        state
    }

    /// Execute one physics step with the given 4D action. Returns (state, `action_cost`).
    fn do_step(&self, action: &[f32]) -> (Vec<f32>, f32) {
        let world = self.world.as_ref().expect("world must exist").clone();

        // Apply motor controls (torque control mode, matching Gymnasium default)
        for (idx, &act) in action.iter().enumerate() {
            let clamped = act.clamp(-1.0, 1.0);
            let speed_factor = if idx % 2 == 0 { SPEED_HIP } else { SPEED_KNEE };
            let motor_speed = speed_factor * clamped.signum();
            let motor_torque = MOTORS_TORQUE * clamped.abs().clamp(0.0, 1.0);

            let mut joint_ref = self.joints[idx].borrow_mut();
            if let JointAsDerivedMut::ERevoluteJoint(rj) = joint_ref.as_derived_mut() {
                rj.set_motor_speed(motor_speed);
                rj.set_max_motor_torque(motor_torque);
            }
        }

        // Step physics
        world.borrow_mut().step(1.0 / FPS, 6 * 30, 2 * 30);

        let state = self.get_state();

        // Compute action cost
        let action_cost: f32 = action
            .iter()
            .map(|a| 0.00035 * MOTORS_TORQUE * a.abs().clamp(0.0, 1.0))
            .sum();

        (state, action_cost)
    }
}

impl Env for BipedalWalkerEnv {
    type Obs = Vec<f32>;
    type Act = Vec<f32>;
    type ObsSpace = BoundedSpace;
    type ActSpace = BoundedSpace;

    fn step(&mut self, action: &Vec<f32>) -> Result<StepResult<Vec<f32>>> {
        if self.world.is_none() {
            return Err(Error::ResetNeeded { method: "step" });
        }

        if action.len() != 4 {
            return Err(Error::InvalidAction {
                reason: format!("expected 4 values, got {}", action.len()),
            });
        }

        let (state, action_cost) = self.do_step(action);

        // Read hull position
        let hull_x = self
            .hull
            .as_ref()
            .expect("hull must exist")
            .borrow()
            .get_position()
            .x;
        self.scroll = hull_x - VIEWPORT_W / SCALE / 5.0;

        // Reward shaping (use f64 for precision, matching Gymnasium)
        let shaping = 5.0f64.mul_add(
            -f64::from(state[0]).abs(),
            130.0 * f64::from(hull_x) / f64::from(SCALE),
        );

        let mut reward = self.prev_shaping.map_or(0.0, |prev| shaping - prev);
        self.prev_shaping = Some(shaping);

        // Subtract motor cost
        reward -= f64::from(action_cost);

        // Check termination
        let game_over = *self.game_over.borrow();
        let terminated = if game_over || hull_x < 0.0 {
            reward = -100.0;
            true
        } else {
            hull_x > (TERRAIN_LENGTH - TERRAIN_GRASS) as f32 * TERRAIN_STEP
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
        self.terrain_bodies.clear();
        self.legs.clear();
        self.joints.clear();
        self.hull = None;
        self.world = None;

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

    fn action_space(&self) -> &BoundedSpace {
        &self.action_space
    }

    fn render_mode(&self) -> &RenderMode {
        &self.render_mode
    }

    fn metadata(&self) -> EnvMetadata {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        EnvMetadata {
            render_fps: Some(FPS as u32),
            ..EnvMetadata::DEFAULT
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::Space;

    #[test]
    fn create_and_reset() {
        let mut env = BipedalWalkerEnv::new(BipedalWalkerConfig::default()).unwrap();
        let r = env.reset(Some(42)).unwrap();
        assert_eq!(r.obs.len(), 24);
    }

    #[test]
    fn create_and_reset_hardcore() {
        let mut env = BipedalWalkerEnv::new(BipedalWalkerConfig {
            hardcore: true,
            ..BipedalWalkerConfig::default()
        })
        .unwrap();
        let r = env.reset(Some(42)).unwrap();
        assert_eq!(r.obs.len(), 24);
    }

    #[test]
    fn step_random_actions() {
        let mut env = BipedalWalkerEnv::new(BipedalWalkerConfig::default()).unwrap();
        env.reset(Some(42)).unwrap();

        let mut rng = rng::create_rng(Some(0));
        for _ in 0..50 {
            let action = env.action_space().sample(&mut rng);
            let s = env.step(&action).unwrap();
            assert_eq!(s.obs.len(), 24);
            if s.terminated {
                break;
            }
        }
    }

    #[test]
    fn step_before_reset_errors() {
        let mut env = BipedalWalkerEnv::new(BipedalWalkerConfig::default()).unwrap();
        assert!(env.step(&vec![0.0; 4]).is_err());
    }

    #[test]
    fn invalid_action_length_errors() {
        let mut env = BipedalWalkerEnv::new(BipedalWalkerConfig::default()).unwrap();
        env.reset(Some(0)).unwrap();
        assert!(env.step(&vec![0.0; 3]).is_err());
    }

    #[test]
    fn seed_determinism() {
        let mut env1 = BipedalWalkerEnv::new(BipedalWalkerConfig::default()).unwrap();
        let mut env2 = BipedalWalkerEnv::new(BipedalWalkerConfig::default()).unwrap();
        let o1 = env1.reset(Some(123)).unwrap().obs;
        let o2 = env2.reset(Some(123)).unwrap().obs;
        assert_eq!(o1, o2);
    }

    #[test]
    fn episode_terminates() {
        let mut env = BipedalWalkerEnv::new(BipedalWalkerConfig::default()).unwrap();
        env.reset(Some(0)).unwrap();
        let mut done = false;
        for _ in 0..2000 {
            let s = env.step(&vec![0.0; 4]).unwrap();
            if s.terminated {
                done = true;
                break;
            }
        }
        assert!(done, "episode should terminate within 2000 steps");
    }
}
