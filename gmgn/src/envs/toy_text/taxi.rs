//! Taxi environment.
//!
//! Navigate a taxi in a 5×5 grid to pick up a passenger at one of four
//! locations and drop them off at their destination.
//!
//! Mirrors [Gymnasium `Taxi-v3`](https://gymnasium.farama.org/environments/toy_text/taxi/).

use std::collections::HashMap;

use rand::RngExt as _;

use crate::env::{Env, InfoValue, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::{Error, Result};
use crate::rng::{self, Rng};
use crate::space::{Discrete, Space};

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

    /// Map description as bytes (kept for future ANSI rendering).
    #[allow(dead_code)]
    desc: Vec<Vec<u8>>,

    state: Option<i64>,
    last_action: Option<i64>,
    rng: Rng,
    render_mode: RenderMode,

    /// Initial-state distribution.
    initial_state_distrib: Vec<f64>,
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

    /// Sample a state from a categorical distribution.
    #[allow(clippy::cast_possible_wrap)]
    fn categorical_sample(probs: &[f64], rng: &mut Rng) -> i64 {
        let r: f64 = rng.random_range(0.0..1.0);
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r < cum {
                return i as i64;
            }
        }
        (probs.len() - 1) as i64
    }

    /// Create a new taxi environment.
    ///
    /// # Errors
    ///
    /// Returns an error if the space cannot be constructed (should not happen).
    #[allow(clippy::cast_possible_wrap, clippy::needless_pass_by_value)]
    pub fn new(config: TaxiConfig) -> Result<Self> {
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

        Ok(Self {
            observation_space: Discrete::new(NUM_STATES),
            action_space: Discrete::new(NUM_ACTIONS),
            transitions,
            desc,
            state: None,
            last_action: None,
            rng: rng::create_rng(None),
            render_mode: config.render_mode,
            initial_state_distrib,
        })
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
        let idx = Self::categorical_sample(&probs, &mut self.rng) as usize;
        let (p, ns, reward, terminated) = trans[idx];

        self.state = Some(ns);
        self.last_action = Some(*action);

        let mut info = HashMap::new();
        info.insert("prob".to_owned(), InfoValue::Float(p));

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

        let s = Self::categorical_sample(&self.initial_state_distrib, &mut self.rng);
        self.state = Some(s);
        self.last_action = None;

        let mut info = HashMap::new();
        info.insert("prob".to_owned(), InfoValue::Float(1.0));

        Ok(ResetResult { obs: s, info })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn render(&mut self) -> Result<RenderFrame> {
        match self.render_mode {
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
            _ => Ok(RenderFrame::None),
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
        TaxiEnv::new(TaxiConfig::default()).unwrap()
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
