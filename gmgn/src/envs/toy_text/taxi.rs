//! Taxi environment.
//!
//! Navigate a taxi in a 5×5 grid to pick up a passenger at one of four
//! locations and drop them off at their destination.
//!
//! Mirrors [Gymnasium `Taxi-v3`](https://gymnasium.farama.org/environments/toy_text/taxi/).

use std::collections::HashMap;

use crate::env::{Env, InfoValue, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::{Error, Result};
#[cfg(feature = "render")]
use crate::render::{Canvas, RenderWindow, sprites::TaxiSprites};
use crate::rng::{self, Rng};
use crate::space::{Discrete, Space};

#[cfg(feature = "render")]
const CELL_SIZE: u32 = 80;
#[cfg(feature = "render")]
const RENDER_FPS: usize = 4;

/// Sample an index from a categorical probability distribution.
fn categorical_sample(probs: &[f64], rng: &mut Rng) -> usize {
    use rand::RngExt as _;
    let r: f64 = rng.random_range(0.0..1.0);
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    probs.len() - 1
}

// The 5×5 grid map (Gymnasium default).
const MAP: &[&str] = &[
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
];

// Four named locations: Red(0,0), Green(0,4), Yellow(4,0), Blue(4,3).
const LOCS: [(usize, usize); 4] = [(0, 0), (0, 4), (4, 0), (4, 3)];

const NUM_ROWS: usize = 5;
const NUM_COLS: usize = 5;
const NUM_STATES: u64 = 500; // 5*5*5*4
const NUM_ACTIONS: u64 = 6;

/// A single transition entry: `(probability, next_state, reward, terminated)`.
type Transition = (f64, i64, f64, bool);

/// Configuration for [`TaxiEnv`].
#[derive(Debug, Clone, Copy)]
pub struct TaxiConfig {
    /// The render mode for this environment.
    pub render_mode: RenderMode,
}

impl Default for TaxiConfig {
    fn default() -> Self {
        Self {
            render_mode: RenderMode::None,
        }
    }
}

/// The Taxi environment.
///
/// # Action Space
///
/// `Discrete(6)`: 0 = South, 1 = North, 2 = East, 3 = West,
/// 4 = Pickup, 5 = Dropoff.
///
/// # Observation Space
///
/// `Discrete(500)`: encoded state
/// `((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest_idx`.
///
/// # Rewards
///
/// - −1 per step (default).
/// - +20 for successful dropoff.
/// - −10 for illegal pickup/dropoff.
///
/// # Episode End
///
/// - **Termination**: passenger is dropped off (at any location; +20
///   only if correct).
/// - **Truncation**: handled externally by a
///   [`TimeLimit`](crate::wrappers::TimeLimit) wrapper (200 steps).
pub struct TaxiEnv {
    action_space: Discrete,
    observation_space: Discrete,

    /// Transition table: `P[state][action] = Vec<Transition>`.
    transitions: Vec<Vec<Vec<Transition>>>,

    /// Map description as bytes (used for wall detection in rendering).
    desc: Vec<Vec<u8>>,

    state: Option<i64>,
    last_action: Option<i64>,
    rng: Rng,
    render_mode: RenderMode,

    /// Initial-state distribution.
    initial_state_distrib: Vec<f64>,

    #[cfg(feature = "render")]
    canvas: Option<Canvas>,
    #[cfg(feature = "render")]
    window: Option<RenderWindow>,
    #[cfg(feature = "render")]
    sprites: Option<TaxiSprites>,
}

impl std::fmt::Debug for TaxiEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaxiEnv")
            .field("state", &self.state)
            .field("render_mode", &self.render_mode)
            .finish_non_exhaustive()
    }
}

impl TaxiEnv {
    /// Encode a taxi state into a single integer.
    #[allow(clippy::cast_possible_wrap)]
    const fn encode(taxi_row: usize, taxi_col: usize, pass_loc: usize, dest_idx: usize) -> i64 {
        (((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest_idx) as i64
    }

    /// Decode a state integer into `(taxi_row, taxi_col, pass_loc, dest_idx)`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    const fn decode(mut i: i64) -> (usize, usize, usize, usize) {
        let dest_idx = (i % 4) as usize;
        i /= 4;
        let pass_loc = (i % 5) as usize;
        i /= 5;
        let taxi_col = (i % 5) as usize;
        i /= 5;
        let taxi_row = i as usize;
        (taxi_row, taxi_col, pass_loc, dest_idx)
    }

    /// Compute an action mask for the given state.
    ///
    /// Returns a 6-element array where `1` means the action will cause a
    /// state change and `0` means it is a no-op or illegal.
    /// Mirrors Gymnasium's `TaxiEnv.action_mask`.
    fn action_mask_for(desc: &[Vec<u8>], state: i64) -> [i64; 6] {
        let (taxi_row, taxi_col, pass_loc, dest_idx) = Self::decode(state);
        let mut mask = [0i64; 6];
        // South.
        if taxi_row < 4 {
            mask[0] = 1;
        }
        // North.
        if taxi_row > 0 {
            mask[1] = 1;
        }
        // East — check wall.
        if taxi_col < 4 && desc[taxi_row + 1][2 * taxi_col + 2] == b':' {
            mask[2] = 1;
        }
        // West — check wall.
        if taxi_col > 0 && desc[taxi_row + 1][2 * taxi_col] == b':' {
            mask[3] = 1;
        }
        // Pickup.
        if pass_loc < 4 && (taxi_row, taxi_col) == LOCS[pass_loc] {
            mask[4] = 1;
        }
        // Dropoff.
        if pass_loc == 4
            && ((taxi_row, taxi_col) == LOCS[dest_idx] || LOCS.contains(&(taxi_row, taxi_col)))
        {
            mask[5] = 1;
        }
        mask
    }

    /// Create a new taxi environment.
    #[allow(clippy::cast_possible_wrap, clippy::needless_pass_by_value)]
    #[must_use]
    pub fn new(config: TaxiConfig) -> Self {
        let desc: Vec<Vec<u8>> = MAP.iter().map(|s| s.as_bytes().to_vec()).collect();

        #[allow(clippy::cast_possible_truncation)]
        let mut initial_state_distrib = vec![0.0; NUM_STATES as usize];
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let mut transitions: Vec<Vec<Vec<Transition>>> =
            vec![vec![Vec::new(); NUM_ACTIONS as usize]; NUM_STATES as usize];

        for row in 0..NUM_ROWS {
            for col in 0..NUM_COLS {
                #[allow(clippy::needless_range_loop)]
                for pass_idx in 0..5_usize {
                    #[allow(clippy::needless_range_loop)]
                    for dest_idx in 0..4_usize {
                        let state = Self::encode(row, col, pass_idx, dest_idx);

                        // Initial states: passenger not in taxi, not at dest.
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        if pass_idx < 4 && pass_idx != dest_idx {
                            initial_state_distrib[state as usize] += 1.0;
                        }

                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        for action in 0..NUM_ACTIONS {
                            let li = &mut transitions[state as usize][action as usize];

                            let taxi_loc = (row, col);
                            let mut new_row = row;
                            let mut new_col = col;
                            let mut new_pass_idx = pass_idx;
                            let mut reward: f64 = -1.0;
                            let mut terminated = false;

                            match action {
                                0 => {
                                    // South.
                                    new_row = (row + 1).min(NUM_ROWS - 1);
                                }
                                1 => {
                                    // North.
                                    new_row = row.saturating_sub(1);
                                }
                                2 => {
                                    // East — check wall.
                                    if desc[1 + row][2 * col + 2] == b':' {
                                        new_col = (col + 1).min(NUM_COLS - 1);
                                    }
                                }
                                3 => {
                                    // West — check wall.
                                    if desc[1 + row][2 * col] == b':' {
                                        new_col = col.saturating_sub(1);
                                    }
                                }
                                4 => {
                                    // Pickup.
                                    if pass_idx < 4 && taxi_loc == LOCS[pass_idx] {
                                        new_pass_idx = 4; // in taxi
                                    } else {
                                        reward = -10.0;
                                    }
                                }
                                5 => {
                                    // Dropoff.
                                    if pass_idx == 4 && taxi_loc == LOCS[dest_idx] {
                                        new_pass_idx = dest_idx;
                                        terminated = true;
                                        reward = 20.0;
                                    } else if pass_idx == 4 {
                                        // Drop at wrong but valid location.
                                        if let Some(loc_idx) =
                                            LOCS.iter().position(|&l| l == taxi_loc)
                                        {
                                            new_pass_idx = loc_idx;
                                        } else {
                                            reward = -10.0;
                                        }
                                    } else {
                                        reward = -10.0;
                                    }
                                }
                                _ => {}
                            }

                            let new_state = Self::encode(new_row, new_col, new_pass_idx, dest_idx);
                            li.push((1.0, new_state, reward, terminated));
                        }
                    }
                }
            }
        }

        // Normalize initial state distribution.
        let sum: f64 = initial_state_distrib.iter().sum();
        if sum > 0.0 {
            for p in &mut initial_state_distrib {
                *p /= sum;
            }
        }

        Self {
            observation_space: Discrete::new(NUM_STATES),
            action_space: Discrete::new(NUM_ACTIONS),
            transitions,
            desc,
            state: None,
            last_action: None,
            rng: rng::create_rng(None),
            render_mode: config.render_mode,
            initial_state_distrib,
            #[cfg(feature = "render")]
            canvas: None,
            #[cfg(feature = "render")]
            window: None,
            #[cfg(feature = "render")]
            sprites: None,
        }
    }

    /// Render the taxi grid using PNG sprites, matching Gymnasium's official look.
    ///
    /// The desc map is 7×11 chars; each cell in the 5×5 grid maps to a desc
    /// coordinate via `get_surf_loc`.  Mirrors Gymnasium's `_render_gui`.
    #[cfg(feature = "render")]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        clippy::many_single_char_names
    )]
    fn render_pixels(&mut self) -> Result<RenderFrame> {
        if self.state.is_none() {
            return Err(Error::ResetNeeded { method: "render" });
        }
        let s = self.state.expect("checked above");
        let (taxi_row, taxi_col, pass_idx, dest_idx) = Self::decode(s);

        // Gymnasium uses window size = desc shape × cell_size.
        let desc_rows = self.desc.len();
        let desc_cols = self.desc[0].len();
        let win_w = desc_cols as u32 * CELL_SIZE;
        let win_h = desc_rows as u32 * CELL_SIZE;

        let sprites = self
            .sprites
            .get_or_insert_with(|| TaxiSprites::new(CELL_SIZE, CELL_SIZE));

        let canvas = self.canvas.get_or_insert_with(|| Canvas::new(win_w, win_h));
        canvas.clear(tiny_skia::Color::WHITE);

        // Draw desc-grid background and wall medians.
        for y in 0..desc_rows {
            for x in 0..desc_cols {
                let px = (x as u32 * CELL_SIZE) as i32;
                let py = (y as u32 * CELL_SIZE) as i32;
                canvas.blit(px, py, &sprites.bg);

                let ch = self.desc[y][x];
                if ch == b'|' {
                    // Vertical wall median.
                    let img = if y == 0 || self.desc[y - 1][x] != b'|' {
                        &sprites.median_vert[0] // top cap
                    } else if y == desc_rows - 1 || self.desc[y + 1][x] != b'|' {
                        &sprites.median_vert[2] // bottom cap
                    } else {
                        &sprites.median_vert[1] // middle
                    };
                    canvas.blit(px, py, img);
                } else if ch == b'-' {
                    // Horizontal wall median.
                    let img = if x == 0 || self.desc[y][x - 1] != b'-' {
                        &sprites.median_horiz[0] // left cap
                    } else if x == desc_cols - 1 || self.desc[y][x + 1] != b'-' {
                        &sprites.median_horiz[2] // right cap
                    } else {
                        &sprites.median_horiz[1] // middle
                    };
                    canvas.blit(px, py, img);
                }
            }
        }

        // Location color overlays (R=red, G=green, Y=yellow, B=blue).
        let loc_colors: [(u8, u8, u8); 4] = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)];
        for (loc, color) in LOCS.iter().zip(loc_colors.iter()) {
            let (lr, lc) = Self::surf_loc(*loc);
            // Create a semi-transparent color overlay.
            let overlay_color = tiny_skia::Color::from_rgba8(color.0, color.1, color.2, 128);
            canvas.fill_rect(
                lr as f32,
                (lc + 10) as f32,
                CELL_SIZE as f32,
                CELL_SIZE as f32,
                overlay_color,
            );
        }

        // Draw passenger sprite at pickup location.
        if pass_idx < 4 {
            let (px, py) = Self::surf_loc(LOCS[pass_idx]);
            canvas.blit(px, py, &sprites.passenger);
        }

        // Taxi orientation from last action: [south=0, north=1, east=2, west=3].
        let taxi_orient = match self.last_action {
            Some(a @ 0..=3) => a as usize,
            _ => 0,
        };
        let (dest_x, dest_y) = Self::surf_loc(LOCS[dest_idx]);
        let (taxi_x, taxi_y) = Self::surf_loc((taxi_row, taxi_col));

        // Blit destination and taxi with depth-correct ordering.
        let hotel_y = dest_y - CELL_SIZE as i32 / 2;
        if dest_y <= taxi_y {
            canvas.blit_with_alpha(dest_x, hotel_y, &sprites.hotel, 170);
            canvas.blit(taxi_x, taxi_y, &sprites.cab[taxi_orient]);
        } else {
            canvas.blit(taxi_x, taxi_y, &sprites.cab[taxi_orient]);
            canvas.blit_with_alpha(dest_x, hotel_y, &sprites.hotel, 170);
        }

        match self.render_mode {
            RenderMode::Human => {
                let window = self.window.get_or_insert_with(|| {
                    RenderWindow::new(
                        "Taxi \u{2014} gmgn",
                        win_w as usize,
                        win_h as usize,
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
                    width: win_w,
                    height: win_h,
                    data: rgb,
                })
            }
            _ => Ok(RenderFrame::None),
        }
    }

    /// Convert a grid `(row, col)` to pixel `(x, y)` matching Gymnasium's `get_surf_loc`.
    #[cfg(feature = "render")]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap
    )]
    const fn surf_loc(map_loc: (usize, usize)) -> (i32, i32) {
        let x = (map_loc.1 * 2 + 1) * CELL_SIZE as usize;
        let y = (map_loc.0 + 1) * CELL_SIZE as usize;
        (x as i32, y as i32)
    }
}

impl Env for TaxiEnv {
    type Obs = i64;
    type Act = i64;
    type ObsSpace = Discrete;
    type ActSpace = Discrete;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn step(&mut self, action: &i64) -> Result<StepResult<i64>> {
        if self.state.is_none() {
            return Err(Error::ResetNeeded { method: "step" });
        }
        if !self.action_space.contains(action) {
            return Err(Error::InvalidAction {
                reason: format!("action {action} not in {{0..5}}"),
            });
        }

        let s = self.state.expect("checked above") as usize;
        let a = *action as usize;
        let trans = &self.transitions[s][a];

        let probs: Vec<f64> = trans.iter().map(|t| t.0).collect();
        let idx = categorical_sample(&probs, &mut self.rng);
        let (p, ns, reward, terminated) = trans[idx];

        self.state = Some(ns);
        self.last_action = Some(*action);

        let mask = Self::action_mask_for(&self.desc, ns);
        let mut info = HashMap::new();
        info.insert("prob".to_owned(), InfoValue::Float(p));
        info.insert("action_mask".to_owned(), InfoValue::IntArray(mask.to_vec()));

        Ok(StepResult {
            obs: ns,
            reward,
            terminated,
            truncated: false,
            info,
        })
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<i64>> {
        if let Some(s) = seed {
            self.rng = rng::create_rng(Some(s));
        }

        #[allow(clippy::cast_possible_wrap)]
        let s = categorical_sample(&self.initial_state_distrib, &mut self.rng) as i64;
        self.state = Some(s);
        self.last_action = None;

        let mask = Self::action_mask_for(&self.desc, s);
        let mut info = HashMap::new();
        info.insert("prob".to_owned(), InfoValue::Float(1.0));
        info.insert("action_mask".to_owned(), InfoValue::IntArray(mask.to_vec()));

        Ok(ResetResult { obs: s, info })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn render(&mut self) -> Result<RenderFrame> {
        match self.render_mode {
            RenderMode::None => Ok(RenderFrame::None),
            RenderMode::Ansi => {
                if self.state.is_none() {
                    return Err(Error::ResetNeeded { method: "render" });
                }
                let s = self.state.expect("checked above");
                let (taxi_row, taxi_col, pass_idx, dest_idx) = Self::decode(s);

                let grid: Vec<Vec<char>> = MAP.iter().map(|row| row.chars().collect()).collect();

                // Highlight taxi position.
                let map_r = taxi_row + 1;
                let map_c = 2 * taxi_col + 1;

                let mut lines = Vec::new();
                if let Some(a) = self.last_action {
                    let dir = match a {
                        0 => "South",
                        1 => "North",
                        2 => "East",
                        3 => "West",
                        4 => "Pickup",
                        5 => "Dropoff",
                        _ => "?",
                    };
                    lines.push(format!("  ({dir})"));
                }

                for (r, row) in grid.iter().enumerate() {
                    let line: String = row
                        .iter()
                        .enumerate()
                        .map(|(c, &ch)| {
                            if r == map_r && c == map_c {
                                if pass_idx == 4 {
                                    // Passenger in taxi — show yellow taxi.
                                    '\u{1F695}'
                                } else {
                                    '\u{1F697}'
                                }
                            } else {
                                ch
                            }
                        })
                        .collect();
                    lines.push(line);
                }

                let loc_names = ["Red", "Green", "Yellow", "Blue"];
                let pass_str = if pass_idx == 4 {
                    "In Taxi".to_owned()
                } else {
                    loc_names[pass_idx].to_owned()
                };
                lines.push(format!(
                    "Passenger: {pass_str}, Destination: {}",
                    loc_names[dest_idx]
                ));

                Ok(RenderFrame::Ansi(lines.join("\n")))
            }
            #[cfg(feature = "render")]
            RenderMode::Human | RenderMode::RgbArray => self.render_pixels(),
            #[cfg(not(feature = "render"))]
            _ => Err(Error::UnsupportedRenderMode {
                mode: format!("{:?}", self.render_mode),
            }),
        }
    }

    fn observation_space(&self) -> &Discrete {
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

    fn make_env() -> TaxiEnv {
        TaxiEnv::new(TaxiConfig::default())
    }

    #[test]
    fn reset_produces_valid_observation() {
        let mut env = make_env();
        let r = env.reset(Some(42)).unwrap();
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
        assert!(env.step(&99).is_err());
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn encode_decode_roundtrip() {
        for row in 0..5 {
            for col in 0..5 {
                for pass in 0..5 {
                    for dest in 0..4 {
                        let s = TaxiEnv::encode(row, col, pass, dest);
                        let (r, c, p, d) = TaxiEnv::decode(s);
                        assert_eq!((r, c, p, d), (row, col, pass, dest));
                    }
                }
            }
        }
    }

    #[test]
    fn illegal_pickup_gives_penalty() {
        let mut env = make_env();
        env.reset(Some(42)).unwrap();
        // State where passenger is NOT at taxi → pickup should give -10.
        // Force a known state: taxi at (0,0), passenger at Green(1), dest Blue(3).
        let s = TaxiEnv::encode(2, 2, 1, 3); // taxi at center, pass at Green
        env.state = Some(s);
        let r = env.step(&4).unwrap(); // pickup
        assert!((r.reward - (-10.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn successful_dropoff_terminates() {
        let mut env = make_env();
        env.reset(Some(0)).unwrap();
        // Place taxi at Red(0,0) with passenger in taxi, dest = Red(0).
        let s = TaxiEnv::encode(0, 0, 4, 0);
        env.state = Some(s);
        let r = env.step(&5).unwrap(); // dropoff
        assert!(r.terminated);
        assert!((r.reward - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn deterministic_with_seed() {
        let mut e1 = make_env();
        let mut e2 = make_env();

        let r1 = e1.reset(Some(99)).unwrap();
        let r2 = e2.reset(Some(99)).unwrap();
        assert_eq!(r1.obs, r2.obs);

        let s1 = e1.step(&0).unwrap();
        let s2 = e2.step(&0).unwrap();
        assert_eq!(s1.obs, s2.obs);
    }

    #[test]
    fn reset_includes_action_mask() {
        let mut env = make_env();
        let r = env.reset(Some(42)).unwrap();
        let mask = r.info.get("action_mask").expect("action_mask missing");
        match mask {
            InfoValue::IntArray(v) => {
                assert_eq!(v.len(), 6);
                // All values should be 0 or 1.
                for &m in v {
                    assert!(m == 0 || m == 1, "mask value out of range: {m}");
                }
            }
            _ => panic!("expected IntArray for action_mask"),
        }
    }

    #[test]
    fn step_includes_action_mask() {
        let mut env = make_env();
        env.reset(Some(42)).unwrap();
        let r = env.step(&0).unwrap();
        let mask = r.info.get("action_mask").expect("action_mask missing");
        match mask {
            InfoValue::IntArray(v) => assert_eq!(v.len(), 6),
            _ => panic!("expected IntArray for action_mask"),
        }
    }

    #[test]
    fn action_mask_pickup_at_correct_location() {
        let env = make_env();
        // Taxi at Red(0,0), passenger at Red(0), dest Blue(3).
        let s = TaxiEnv::encode(0, 0, 0, 3);
        let mask = TaxiEnv::action_mask_for(&env.desc, s);
        // Pickup should be allowed (mask[4] == 1).
        assert_eq!(mask[4], 1);
        // Dropoff should NOT be allowed (passenger not in taxi).
        assert_eq!(mask[5], 0);
    }

    #[test]
    fn action_mask_dropoff_at_destination() {
        let env = make_env();
        // Taxi at Blue(4,3), passenger in taxi(4), dest Blue(3).
        let s = TaxiEnv::encode(4, 3, 4, 3);
        let mask = TaxiEnv::action_mask_for(&env.desc, s);
        // Dropoff should be allowed.
        assert_eq!(mask[5], 1);
        // Pickup should NOT be allowed (pass already in taxi).
        assert_eq!(mask[4], 0);
    }

    #[test]
    fn all_500_states_reachable_in_transitions() {
        let env = make_env();
        // Every state should have transitions for all 6 actions.
        for s in 0..500 {
            for a in 0..6 {
                assert!(
                    !env.transitions[s][a].is_empty(),
                    "state {s} action {a} has no transitions"
                );
            }
        }
    }
}
