//! Frozen lake environment.
//!
//! The player navigates a grid of frozen tiles (`F`), holes (`H`), a start
//! tile (`S`), and a goal tile (`G`).  When `is_slippery` is enabled the
//! agent moves in the intended direction with probability `1/3` and
//! perpendicular with probability `1/3` each.
//!
//! Mirrors [Gymnasium `FrozenLake-v1`](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).

use std::collections::HashMap;

use crate::env::{Env, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::{Error, Result};
#[cfg(feature = "render")]
use crate::render::{Canvas, RenderWindow, sprites::FrozenLakeSprites};
use crate::rng::{self, Rng};
use crate::space::{Discrete, Space};

#[cfg(feature = "render")]
const CELL_SIZE: u32 = 64;
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

// Actions.
const LEFT: i64 = 0;
const DOWN: i64 = 1;
const RIGHT: i64 = 2;
const UP: i64 = 3;

/// Pre-defined maps shipped with Gymnasium.
pub const MAP_4X4: &[&str] = &["SFFF", "FHFH", "FFFH", "HFFG"];

/// 8×8 pre-defined map.
pub const MAP_8X8: &[&str] = &[
    "SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG",
];

/// Check whether there is a path from `S` (top-left) to `G` using DFS.
///
/// Mirrors Gymnasium's `is_valid` helper.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn is_valid_map(board: &[Vec<u8>], size: usize) -> bool {
    let mut discovered = vec![vec![false; size]; size];
    let mut stack = vec![(0usize, 0usize)];
    while let Some((r, c)) = stack.pop() {
        if discovered[r][c] {
            continue;
        }
        discovered[r][c] = true;
        for (dr, dc) in [(1i32, 0i32), (0, 1), (-1, 0), (0, -1)] {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nr >= size as i32 || nc < 0 || nc >= size as i32 {
                continue;
            }
            let (nr, nc) = (nr as usize, nc as usize);
            if board[nr][nc] == b'G' {
                return true;
            }
            if board[nr][nc] != b'H' {
                stack.push((nr, nc));
            }
        }
    }
    false
}

/// Generate a random valid frozen-lake map.
///
/// Keeps generating random boards until one with a valid path from `S` to `G`
/// is found.  Mirrors Gymnasium's `generate_random_map`.
///
/// # Arguments
///
/// - `size` — Side length of the square grid (default 8).
/// - `p` — Probability that a tile is frozen (`F`) rather than a hole (`H`).
/// - `seed` — Optional RNG seed for reproducibility.
///
/// # Panics
///
/// Panics if the generated board contains non-ASCII bytes (should never
/// happen since only `S`, `F`, `H`, `G` are used).
///
/// # Examples
///
/// ```
/// let map = gmgn::envs::toy_text::generate_random_map(8, 0.8, Some(42));
/// assert_eq!(map.len(), 8);
/// assert!(map[0].starts_with('S'));
/// assert!(map[7].ends_with('G'));
/// ```
#[must_use]
pub fn generate_random_map(size: usize, p: f64, seed: Option<u64>) -> Vec<String> {
    use rand::RngExt as _;

    let mut rng = rng::create_rng(seed);
    let p = p.min(1.0);

    loop {
        let mut board: Vec<Vec<u8>> = (0..size)
            .map(|_| {
                (0..size)
                    .map(|_| {
                        if rng.random_range(0.0..1.0) < p {
                            b'F'
                        } else {
                            b'H'
                        }
                    })
                    .collect()
            })
            .collect();
        board[0][0] = b'S';
        board[size - 1][size - 1] = b'G';

        if is_valid_map(&board, size) {
            return board
                .iter()
                .map(|row| String::from_utf8(row.clone()).expect("ASCII"))
                .collect();
        }
    }
}

/// A single transition entry: `(probability, next_state, reward, terminated)`.
type Transition = (f64, i64, f64, bool);

/// Configuration for [`FrozenLakeEnv`].
#[derive(Debug, Clone)]
pub struct FrozenLakeConfig {
    /// Grid description.  Each string is one row; valid tiles are
    /// `S` (start), `F` (frozen), `H` (hole), `G` (goal).
    pub desc: Vec<String>,
    /// Whether the lake is slippery (stochastic transitions).
    pub is_slippery: bool,
    /// Render mode.
    pub render_mode: RenderMode,
}

impl Default for FrozenLakeConfig {
    fn default() -> Self {
        Self {
            desc: MAP_4X4.iter().map(|s| (*s).to_owned()).collect(),
            is_slippery: true,
            render_mode: RenderMode::None,
        }
    }
}

/// Frozen lake environment.
///
/// # Action Space
///
/// `Discrete(4)`: 0 = Left, 1 = Down, 2 = Right, 3 = Up.
///
/// # Observation Space
///
/// `Discrete(nrow × ncol)`: current cell index (`row * ncol + col`).
///
/// # Rewards
///
/// +1 for reaching the goal, 0 otherwise.
///
/// # Episode End
///
/// - **Termination**: the player reaches the goal (`G`) or falls into a
///   hole (`H`).
/// - **Truncation**: handled externally by a
///   [`TimeLimit`](crate::wrappers::TimeLimit) wrapper.
pub struct FrozenLakeEnv {
    action_space: Discrete,
    observation_space: Discrete,

    /// Transition table: `P[state][action] = Vec<Transition>`.
    transitions: Vec<Vec<Vec<Transition>>>,

    nrow: usize,
    ncol: usize,
    desc: Vec<Vec<u8>>,

    state: Option<i64>,
    last_action: Option<i64>,
    rng: Rng,
    render_mode: RenderMode,

    /// Initial-state distribution (uniform over `S` tiles).
    initial_state_distrib: Vec<f64>,

    #[cfg(feature = "render")]
    canvas: Option<Canvas>,
    #[cfg(feature = "render")]
    window: Option<RenderWindow>,
    #[cfg(feature = "render")]
    sprites: Option<FrozenLakeSprites>,
}

impl std::fmt::Debug for FrozenLakeEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrozenLakeEnv")
            .field("nrow", &self.nrow)
            .field("ncol", &self.ncol)
            .field("state", &self.state)
            .field("render_mode", &self.render_mode)
            .finish_non_exhaustive()
    }
}

impl FrozenLakeEnv {
    /// Create a new frozen lake environment.
    ///
    /// # Errors
    ///
    /// Returns an error if the map description is empty or inconsistent.
    #[allow(clippy::cast_possible_wrap, clippy::needless_pass_by_value)]
    pub fn new(config: FrozenLakeConfig) -> Result<Self> {
        let desc: Vec<Vec<u8>> = config
            .desc
            .iter()
            .map(|row| row.as_bytes().to_vec())
            .collect();

        if desc.is_empty() {
            return Err(Error::InvalidSpace {
                reason: "map description is empty".into(),
            });
        }

        let nrow = desc.len();
        let ncol = desc[0].len();
        #[allow(clippy::cast_possible_truncation)]
        let n_states = (nrow * ncol) as u64;
        let n_actions = 4_u64;

        // Build initial-state distribution (uniform over 'S' tiles).
        #[allow(clippy::cast_possible_truncation)]
        let mut initial_state_distrib = vec![0.0; n_states as usize];
        for (r, row) in desc.iter().enumerate() {
            for (c, &tile) in row.iter().enumerate() {
                if tile == b'S' {
                    initial_state_distrib[r * ncol + c] = 1.0;
                }
            }
        }
        let sum: f64 = initial_state_distrib.iter().sum();
        if sum > 0.0 {
            for p in &mut initial_state_distrib {
                *p /= sum;
            }
        }

        // Helper: move in direction `a` from (row, col).
        let inc = |row: usize, col: usize, a: i64| -> (usize, usize) {
            match a {
                LEFT => (row, col.saturating_sub(1)),
                DOWN => (row.min(nrow - 2) + 1, col),
                RIGHT => (row, (col + 1).min(ncol - 1)),
                UP => (row.saturating_sub(1), col),
                _ => (row, col),
            }
        };

        // Build transition table P[s][a].
        #[allow(clippy::cast_possible_truncation)]
        let mut transitions: Vec<Vec<Vec<Transition>>> =
            vec![vec![Vec::new(); n_actions as usize]; n_states as usize];

        for r in 0..nrow {
            for c in 0..ncol {
                let s = (r * ncol + c) as i64;
                let tile = desc[r][c];

                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                for a in 0..n_actions {
                    let li = &mut transitions[s as usize][a as usize];

                    if tile == b'G' || tile == b'H' {
                        // Terminal absorbing state.
                        li.push((1.0, s, 0.0, true));
                    } else if config.is_slippery {
                        // Slippery: intended direction + two perpendicular.
                        #[allow(clippy::cast_possible_wrap)]
                        for b_offset in [-1_i64, 0, 1] {
                            let b = (a as i64 + b_offset).rem_euclid(4) as usize;
                            let (nr, nc2) = inc(r, c, b as i64);
                            let ns = (nr * ncol + nc2) as i64;
                            let new_tile = desc[nr][nc2];
                            let terminated = new_tile == b'G' || new_tile == b'H';
                            let reward = if new_tile == b'G' { 1.0 } else { 0.0 };
                            li.push((1.0 / 3.0, ns, reward, terminated));
                        }
                    } else {
                        // Deterministic.
                        let (nr, nc2) = inc(r, c, a.cast_signed());
                        let ns = (nr * ncol + nc2) as i64;
                        let new_tile = desc[nr][nc2];
                        let terminated = new_tile == b'G' || new_tile == b'H';
                        let reward = if new_tile == b'G' { 1.0 } else { 0.0 };
                        li.push((1.0, ns, reward, terminated));
                    }
                }
            }
        }

        Ok(Self {
            observation_space: Discrete::new(n_states),
            action_space: Discrete::new(n_actions),
            transitions,
            nrow,
            ncol,
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
        })
    }

    /// Render the grid using PNG sprites, matching Gymnasium's official look.
    ///
    /// Uses ice/hole/goal/stool/elf PNG assets scaled to cell size.
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
        let s = self.state.expect("checked above") as usize;
        let cur_row = s / self.ncol;
        let cur_col = s % self.ncol;

        let w = (self.ncol as u32) * CELL_SIZE;
        let h = (self.nrow as u32) * CELL_SIZE;

        let sprites = self
            .sprites
            .get_or_insert_with(|| FrozenLakeSprites::new(CELL_SIZE, CELL_SIZE));

        let canvas = self.canvas.get_or_insert_with(|| Canvas::new(w, h));
        canvas.clear(tiny_skia::Color::WHITE);

        let grid_color = tiny_skia::Color::from_rgba8(180, 200, 230, 255);

        // Draw tiles — mirrors Gymnasium's _render_gui exactly.
        for r in 0..self.nrow {
            for c in 0..self.ncol {
                let px = (c as u32 * CELL_SIZE) as i32;
                let py = (r as u32 * CELL_SIZE) as i32;
                let tile = self.desc[r][c];

                // Base: ice background on every cell.
                canvas.blit(px, py, &sprites.ice);

                // Overlay per tile type.
                match tile {
                    b'H' => canvas.blit(px, py, &sprites.hole),
                    b'G' => canvas.blit(px, py, &sprites.goal),
                    b'S' => canvas.blit(px, py, &sprites.stool),
                    _ => {}
                }

                // Grid lines (1 px border like Gymnasium).
                let fx = px as f32;
                let fy = py as f32;
                let cs = CELL_SIZE as f32;
                canvas.stroke_line(fx, fy, fx + cs, fy, 1.0, grid_color);
                canvas.stroke_line(fx + cs, fy, fx + cs, fy + cs, 1.0, grid_color);
                canvas.stroke_line(fx + cs, fy + cs, fx, fy + cs, 1.0, grid_color);
                canvas.stroke_line(fx, fy + cs, fx, fy, 1.0, grid_color);
            }
        }

        // Draw the elf (player).
        let bot_x = (cur_col as u32 * CELL_SIZE) as i32;
        let bot_y = (cur_row as u32 * CELL_SIZE) as i32;
        let last_action = self.last_action.unwrap_or(DOWN) as usize;
        let tile = self.desc[cur_row][cur_col];

        if tile == b'H' {
            // Elf fell into a hole — show cracked hole.
            canvas.blit(bot_x, bot_y, &sprites.cracked_hole);
        } else {
            // Elf sprite oriented by last action: [left=0, down=1, right=2, up=3].
            let elf_idx = last_action.min(3);
            canvas.blit(bot_x, bot_y, &sprites.elf[elf_idx]);
        }

        match self.render_mode {
            RenderMode::Human => {
                let window = self.window.get_or_insert_with(|| {
                    RenderWindow::new(
                        "FrozenLake \u{2014} gmgn",
                        w as usize,
                        h as usize,
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
                    width: w,
                    height: h,
                    data: rgb,
                })
            }
            _ => Ok(RenderFrame::None),
        }
    }
}

impl Env for FrozenLakeEnv {
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
                reason: format!("action {action} not in {{0..{}}}", self.action_space.n - 1),
            });
        }

        let s = self.state.expect("checked above") as usize;
        let a = *action as usize;
        let trans = &self.transitions[s][a];

        // Sample from transition distribution.
        let probs: Vec<f64> = trans.iter().map(|t| t.0).collect();
        let idx = categorical_sample(&probs, &mut self.rng);
        let (_p, ns, reward, terminated) = trans[idx];

        self.state = Some(ns);
        self.last_action = Some(*action);

        let mut info = HashMap::new();
        info.insert(
            "prob".to_owned(),
            crate::env::InfoValue::Float(trans[idx].0),
        );

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

        let mut info = HashMap::new();
        info.insert("prob".to_owned(), crate::env::InfoValue::Float(1.0));

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
                let s = self.state.expect("checked above") as usize;
                let row = s / self.ncol;
                let col = s % self.ncol;

                let mut lines = Vec::new();
                if let Some(a) = self.last_action {
                    let dir = match a {
                        LEFT => "Left",
                        DOWN => "Down",
                        RIGHT => "Right",
                        UP => "Up",
                        _ => "?",
                    };
                    lines.push(format!("  ({dir})"));
                }
                for r in 0..self.nrow {
                    let mut line = String::new();
                    for c in 0..self.ncol {
                        let ch = self.desc[r][c] as char;
                        if r == row && c == col {
                            // Highlight current position.
                            line.push('[');
                            line.push(ch);
                            line.push(']');
                        } else {
                            line.push(' ');
                            line.push(ch);
                            line.push(' ');
                        }
                    }
                    lines.push(line);
                }
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
    use crate::space::Space;

    fn make_env(slippery: bool) -> FrozenLakeEnv {
        FrozenLakeEnv::new(FrozenLakeConfig {
            is_slippery: slippery,
            ..FrozenLakeConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn reset_produces_valid_observation() {
        let mut env = make_env(true);
        let r = env.reset(Some(42)).unwrap();
        assert!(env.observation_space().contains(&r.obs));
        // 4x4 default map: start at tile 0.
        assert_eq!(r.obs, 0);
    }

    #[test]
    fn step_without_reset_errors() {
        let mut env = make_env(true);
        assert!(env.step(&0).is_err());
    }

    #[test]
    fn step_invalid_action_errors() {
        let mut env = make_env(true);
        env.reset(Some(0)).unwrap();
        assert!(env.step(&99).is_err());
    }

    #[test]
    fn deterministic_reaches_goal() {
        // Non-slippery: deterministic transitions.
        let mut env = make_env(false);
        env.reset(Some(0)).unwrap();

        // 4x4 map: S F F F  /  F H F H  /  F F F H  /  H F F G
        // Path: (0,0) → (0,1)R → (0,2)R → (1,2)D → (2,2)D → (2,3)R → err
        // Actually: right-right-down-down-right-down
        let actions = [RIGHT, RIGHT, DOWN, DOWN, DOWN, RIGHT];
        let mut terminated = false;
        let mut last_reward = 0.0;
        for &a in &actions {
            if terminated {
                break;
            }
            let r = env.step(&a).unwrap();
            terminated = r.terminated;
            last_reward = r.reward;
        }
        assert!(terminated);
        assert!((last_reward - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn deterministic_hole_terminates() {
        let mut env = make_env(false);
        env.reset(Some(0)).unwrap();
        // Move down from (0,0) to (1,0), then right to (1,1) which is H.
        let r1 = env.step(&DOWN).unwrap();
        assert!(!r1.terminated);
        let r2 = env.step(&RIGHT).unwrap();
        assert!(r2.terminated);
        assert!((r2.reward - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn slippery_transitions_have_3_outcomes() {
        let env = make_env(true);
        // From state 0 (not terminal), each action should have 3 transitions.
        for a in 0..4 {
            assert_eq!(env.transitions[0][a].len(), 3);
        }
    }

    #[test]
    fn terminal_state_is_absorbing() {
        // State 15 = (3,3) = G in 4x4 map.
        let env = make_env(true);
        for a in 0..4 {
            assert_eq!(env.transitions[15][a].len(), 1);
            let (p, ns, _r, t) = env.transitions[15][a][0];
            assert!((p - 1.0).abs() < f64::EPSILON);
            assert_eq!(ns, 15);
            assert!(t);
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let mut e1 = make_env(true);
        let mut e2 = make_env(true);

        let r1 = e1.reset(Some(42)).unwrap();
        let r2 = e2.reset(Some(42)).unwrap();
        assert_eq!(r1.obs, r2.obs);

        let s1 = e1.step(&1).unwrap();
        let s2 = e2.step(&1).unwrap();
        assert_eq!(s1.obs, s2.obs);
        assert!((s1.reward - s2.reward).abs() < f64::EPSILON);
    }

    #[test]
    fn generate_random_map_produces_valid_board() {
        let map = generate_random_map(8, 0.8, Some(42));
        assert_eq!(map.len(), 8);
        for row in &map {
            assert_eq!(row.len(), 8);
        }
        assert!(map[0].starts_with('S'));
        assert!(map[7].ends_with('G'));
        // Verify all tiles are valid characters.
        for row in &map {
            for ch in row.chars() {
                assert!(
                    ch == 'S' || ch == 'F' || ch == 'H' || ch == 'G',
                    "unexpected tile: {ch}"
                );
            }
        }
    }

    #[test]
    fn generate_random_map_deterministic_with_seed() {
        let m1 = generate_random_map(6, 0.7, Some(99));
        let m2 = generate_random_map(6, 0.7, Some(99));
        assert_eq!(m1, m2);
    }

    #[test]
    fn generate_random_map_can_be_used_as_config() {
        let map = generate_random_map(4, 0.9, Some(123));
        let env = FrozenLakeEnv::new(FrozenLakeConfig {
            desc: map,
            is_slippery: false,
            ..FrozenLakeConfig::default()
        })
        .unwrap();
        assert_eq!(env.observation_space().n, 16);
    }

    #[test]
    fn ansi_render() {
        let mut env = FrozenLakeEnv::new(FrozenLakeConfig {
            render_mode: RenderMode::Ansi,
            ..FrozenLakeConfig::default()
        })
        .unwrap();
        env.reset(Some(0)).unwrap();
        let frame = env.render().unwrap();
        match frame {
            RenderFrame::Ansi(text) => assert!(text.contains("[S]")),
            _ => unreachable!("expected Ansi frame"),
        }
    }
}
