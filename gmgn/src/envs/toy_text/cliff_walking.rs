//! Cliff walking environment.
//!
//! A 4×12 grid where the agent starts at `[3, 0]` and must reach `[3, 11]`
//! without stepping on the cliff cells `[3, 1..11]`.  Stepping on a cliff
//! sends the agent back to the start with −100 reward; all other moves
//! cost −1.
//!
//! Mirrors [Gymnasium `CliffWalking-v0`](https://gymnasium.farama.org/environments/toy_text/cliff_walking/).

use std::collections::HashMap;

use crate::env::{Env, InfoValue, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::{Error, Result};
use crate::rng::{self, Rng};
use crate::space::{Discrete, Space};

// Actions.
const UP: i64 = 0;
const RIGHT: i64 = 1;
const DOWN: i64 = 2;
const LEFT: i64 = 3;

const NUM_ROWS: usize = 4;
const NUM_COLS: usize = 12;
const NUM_STATES: u64 = (NUM_ROWS * NUM_COLS) as u64; // 48
const NUM_ACTIONS: u64 = 4;

/// A single transition entry: `(probability, next_state, reward, terminated)`.
type Transition = (f64, i64, f64, bool);

/// Configuration for [`CliffWalkingEnv`].
#[derive(Debug, Clone, Copy)]
pub struct CliffWalkingConfig {
    /// The render mode for this environment.
    pub render_mode: RenderMode,
}

impl Default for CliffWalkingConfig {
    fn default() -> Self {
        Self {
            render_mode: RenderMode::None,
        }
    }
}

/// The cliff walking environment.
///
/// # Action Space
///
/// `Discrete(4)`: 0 = Up, 1 = Right, 2 = Down, 3 = Left.
///
/// # Observation Space
///
/// `Discrete(48)`: current cell index (`row * 12 + col`).
///
/// # Rewards
///
/// −1 per step; −100 when stepping on the cliff (returns to start).
///
/// # Episode End
///
/// - **Termination**: the agent reaches `[3, 11]`.
/// - **Truncation**: handled externally by a
///   [`TimeLimit`](crate::wrappers::TimeLimit) wrapper.
pub struct CliffWalkingEnv {
    action_space: Discrete,
    observation_space: Discrete,

    /// Transition table: `P[state][action] = Vec<Transition>`.
    transitions: Vec<Vec<Vec<Transition>>>,

    state: Option<i64>,
    last_action: Option<i64>,
    rng: Rng,
    render_mode: RenderMode,

    /// Start state index (row=3, col=0 → 36).
    start_state: i64,
}

impl std::fmt::Debug for CliffWalkingEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CliffWalkingEnv")
            .field("state", &self.state)
            .field("render_mode", &self.render_mode)
            .finish_non_exhaustive()
    }
}

impl CliffWalkingEnv {
    /// Create a new cliff walking environment.
    ///
    /// # Errors
    ///
    /// Returns an error if the space cannot be constructed (should not happen).
    #[allow(clippy::cast_possible_wrap, clippy::needless_pass_by_value)]
    pub fn new(config: CliffWalkingConfig) -> Result<Self> {
        let start_state = (3 * NUM_COLS) as i64; // [3, 0] = 36
        let goal_state = (3 * NUM_COLS + NUM_COLS - 1) as i64; // [3, 11] = 47

        // Cliff occupies [3, 1] through [3, 10].
        let is_cliff = |r: usize, c: usize| -> bool { r == 3 && (1..=10).contains(&c) };

        // Movement deltas: UP=[-1,0], RIGHT=[0,1], DOWN=[1,0], LEFT=[0,-1].
        let delta = |action: i64| -> (isize, isize) {
            match action {
                UP => (-1, 0),
                RIGHT => (0, 1),
                DOWN => (1, 0),
                LEFT => (0, -1),
                _ => (0, 0),
            }
        };

        #[allow(clippy::cast_possible_truncation)]
        let mut transitions: Vec<Vec<Vec<Transition>>> =
            vec![vec![Vec::new(); NUM_ACTIONS as usize]; NUM_STATES as usize];

        for r in 0..NUM_ROWS {
            for c in 0..NUM_COLS {
                let s = (r * NUM_COLS + c) as i64;

                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                for a in 0..NUM_ACTIONS as i64 {
                    let li = &mut transitions[s as usize][a as usize];

                    let (dr, dc) = delta(a);
                    let nr = (r as isize + dr).clamp(0, (NUM_ROWS - 1) as isize).cast_unsigned();
                    let nc = (c as isize + dc).clamp(0, (NUM_COLS - 1) as isize).cast_unsigned();
                    let ns = (nr * NUM_COLS + nc) as i64;

                    if is_cliff(nr, nc) {
                        // Cliff → back to start, −100, not terminated.
                        li.push((1.0, start_state, -100.0, false));
                    } else {
                        let terminated = ns == goal_state;
                        li.push((1.0, ns, -1.0, terminated));
                    }
                }
            }
        }

        Ok(Self {
            observation_space: Discrete::new(NUM_STATES),
            action_space: Discrete::new(NUM_ACTIONS),
            transitions,
            state: None,
            last_action: None,
            rng: rng::create_rng(None),
            render_mode: config.render_mode,
            start_state,
        })
    }
}

impl Env for CliffWalkingEnv {
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
                reason: format!("action {action} not in {{0..3}}"),
            });
        }

        let s = self.state.expect("checked above") as usize;
        let a = *action as usize;
        let (p, ns, reward, terminated) = self.transitions[s][a][0]; // deterministic

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

        self.state = Some(self.start_state);
        self.last_action = None;

        let mut info = HashMap::new();
        info.insert("prob".to_owned(), InfoValue::Float(1.0));

        Ok(ResetResult {
            obs: self.start_state,
            info,
        })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn render(&mut self) -> Result<RenderFrame> {
        match self.render_mode {
            RenderMode::Ansi => {
                if self.state.is_none() {
                    return Err(Error::ResetNeeded { method: "render" });
                }
                let s = self.state.expect("checked above") as usize;
                let cur_row = s / NUM_COLS;
                let cur_col = s % NUM_COLS;

                let mut lines = Vec::new();
                if let Some(a) = self.last_action {
                    let dir = match a {
                        UP => "Up",
                        RIGHT => "Right",
                        DOWN => "Down",
                        LEFT => "Left",
                        _ => "?",
                    };
                    lines.push(format!("  ({dir})"));
                }

                for r in 0..NUM_ROWS {
                    let mut line = String::new();
                    for c in 0..NUM_COLS {
                        let ch = if r == cur_row && c == cur_col {
                            'X'
                        } else if r == 3 && c == 0 {
                            'S'
                        } else if r == 3 && c == 11 {
                            'G'
                        } else if r == 3 && (1..=10).contains(&c) {
                            'C'
                        } else {
                            '.'
                        };
                        if r == cur_row && c == cur_col {
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

    fn make_env() -> CliffWalkingEnv {
        CliffWalkingEnv::new(CliffWalkingConfig::default()).unwrap()
    }

    #[test]
    fn reset_starts_at_36() {
        let mut env = make_env();
        let r = env.reset(Some(0)).unwrap();
        assert_eq!(r.obs, 36); // [3, 0]
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
    fn stepping_right_into_cliff_returns_to_start() {
        let mut env = make_env();
        env.reset(Some(0)).unwrap();
        // From [3,0], move right → [3,1] which is cliff → back to [3,0].
        let r = env.step(&RIGHT).unwrap();
        assert_eq!(r.obs, 36); // back to start
        assert!((r.reward - (-100.0)).abs() < f64::EPSILON);
        assert!(!r.terminated);
    }

    #[test]
    fn optimal_path_reaches_goal() {
        let mut env = make_env();
        env.reset(Some(0)).unwrap();
        // Optimal: up, then 11 rights, then down → goal at [3,11]=47.
        let r = env.step(&UP).unwrap(); // [3,0]->[2,0]
        assert_eq!(r.obs, 24);
        for _ in 0..11 {
            env.step(&RIGHT).unwrap();
        }
        // Now at [2,11]=35. Move down to [3,11]=47.
        let r = env.step(&DOWN).unwrap();
        assert_eq!(r.obs, 47);
        assert!(r.terminated);
        assert!((r.reward - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn transitions_are_deterministic() {
        let env = make_env();
        // Each state-action pair should have exactly 1 transition.
        for s in 0..48 {
            for a in 0..4 {
                assert_eq!(env.transitions[s][a].len(), 1);
                assert!((env.transitions[s][a][0].0 - 1.0).abs() < f64::EPSILON);
            }
        }
    }
}
