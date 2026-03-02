//! Blackjack environment.
//!
//! A simple card game where the goal is to beat the dealer by getting a hand
//! value as close to 21 as possible without going over.  Cards are drawn from
//! an infinite deck (with replacement).
//!
//! Mirrors [Gymnasium `Blackjack-v1`](https://gymnasium.farama.org/environments/toy_text/blackjack/).

use std::collections::HashMap;

use rand::RngExt as _;

use crate::env::{Env, InfoValue, RenderFrame, RenderMode, ResetResult, StepResult};
use crate::error::{Error, Result};
#[cfg(feature = "render")]
use crate::render::{Canvas, RenderWindow, sprites::BlackjackSprites};
use crate::rng::{self, Rng};
use crate::space::{Discrete, Space, Tuple3};

#[cfg(feature = "render")]
const SCREEN_W: u32 = 600;
#[cfg(feature = "render")]
const SCREEN_H: u32 = 500;
#[cfg(feature = "render")]
const RENDER_FPS: usize = 4;

// Infinite deck: 1 = Ace, 2-10 = number cards, J/Q/K = 10.
const DECK: [i64; 13] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];

/// Configuration for [`BlackjackEnv`].
#[derive(Debug, Clone, Copy)]
pub struct BlackjackConfig {
    /// Whether to give +1.5 for a natural blackjack win (legacy casino rules).
    pub natural: bool,
    /// Whether to follow Sutton & Barto rules exactly (overrides `natural`).
    pub sab: bool,
    /// The render mode for this environment.
    pub render_mode: RenderMode,
}

impl Default for BlackjackConfig {
    fn default() -> Self {
        Self {
            natural: false,
            sab: true,
            render_mode: RenderMode::None,
        }
    }
}

/// Blackjack observation: `(player_sum, dealer_card, usable_ace)`.
pub type BlackjackObs = (i64, i64, i64);

/// The observation space type for Blackjack.
pub type BlackjackObsSpace = Tuple3<Discrete, Discrete, Discrete>;

/// The Blackjack environment.
///
/// # Action Space
///
/// `Discrete(2)`: 0 = Stick, 1 = Hit.
///
/// # Observation Space
///
/// `Tuple3(Discrete(32), Discrete(11), Discrete(2))`:
/// `(player_sum, dealer_card, usable_ace)`.
///
/// - `player_sum`: 0–31 (effective hand total).
/// - `dealer_card`: 1–10 (dealer's visible card, 1 = Ace).
/// - `usable_ace`: 0 or 1.
///
/// # Rewards
///
/// - Win: +1 (or +1.5 for natural blackjack if `natural=true`).
/// - Lose: −1.
/// - Draw: 0.
///
/// # Episode End
///
/// - **Termination**: the player busts or sticks (dealer then plays).
pub struct BlackjackEnv {
    action_space: Discrete,
    observation_space: BlackjackObsSpace,

    player: Vec<i64>,
    dealer: Vec<i64>,

    natural: bool,
    sab: bool,
    rng: Rng,
    render_mode: RenderMode,

    /// Dealer's top card suit for rendering ('C','D','H','S').
    dealer_top_card_suit: u8,
    /// Dealer's top card value string for rendering ('2'..'9','T','J','Q','K','A').
    dealer_top_card_value: u8,

    #[cfg(feature = "render")]
    canvas: Option<Canvas>,
    #[cfg(feature = "render")]
    window: Option<RenderWindow>,
    #[cfg(feature = "render")]
    sprites: Option<BlackjackSprites>,
}

impl std::fmt::Debug for BlackjackEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlackjackEnv")
            .field("player", &self.player)
            .field("dealer", &self.dealer)
            .field("render_mode", &self.render_mode)
            .finish_non_exhaustive()
    }
}

/// Draw a card from the infinite deck.
fn draw_card(rng: &mut Rng) -> i64 {
    DECK[rng.random_range(0..DECK.len())]
}

/// Draw a two-card starting hand.
fn draw_hand(rng: &mut Rng) -> Vec<i64> {
    vec![draw_card(rng), draw_card(rng)]
}

/// Does this hand have a usable ace?
fn usable_ace(hand: &[i64]) -> bool {
    hand.contains(&1) && hand.iter().sum::<i64>() + 10 <= 21
}

/// Return effective hand total.
fn sum_hand(hand: &[i64]) -> i64 {
    let s: i64 = hand.iter().sum();
    if usable_ace(hand) { s + 10 } else { s }
}

/// Is this hand a bust?
fn is_bust(hand: &[i64]) -> bool {
    sum_hand(hand) > 21
}

/// Score of the hand (0 if bust).
fn score(hand: &[i64]) -> i64 {
    if is_bust(hand) { 0 } else { sum_hand(hand) }
}

/// Is this a natural blackjack (Ace + 10-value card)?
fn is_natural(hand: &[i64]) -> bool {
    hand.len() == 2 && hand.iter().sum::<i64>() == 11 && hand.contains(&1)
}

/// Compare two scores: +1.0 if a > b, −1.0 if a < b, 0.0 if equal.
const fn cmp_scores(a: i64, b: i64) -> f64 {
    if a > b {
        1.0
    } else if a < b {
        -1.0
    } else {
        0.0
    }
}

/// Build the observation tuple from hand state.
fn get_obs(player: &[i64], dealer: &[i64]) -> BlackjackObs {
    (sum_hand(player), dealer[0], i64::from(usable_ace(player)))
}

impl BlackjackEnv {
    /// Create a new blackjack environment.
    #[allow(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn new(config: BlackjackConfig) -> Self {
        Self {
            action_space: Discrete::new(2),
            observation_space: Tuple3::new(Discrete::new(32), Discrete::new(11), Discrete::new(2)),
            player: Vec::new(),
            dealer: Vec::new(),
            natural: config.natural,
            sab: config.sab,
            rng: rng::create_rng(None),
            render_mode: config.render_mode,
            dealer_top_card_suit: b'C',
            dealer_top_card_value: b'2',
            #[cfg(feature = "render")]
            canvas: None,
            #[cfg(feature = "render")]
            window: None,
            #[cfg(feature = "render")]
            sprites: None,
        }
    }

    /// Render the blackjack table using PNG card images, matching Gymnasium's layout.
    ///
    /// Uses 600×500 canvas with green felt, dealer's showing card + hidden card
    /// back, and player sum indicator bar.  Card images from Gymnasium assets.
    #[cfg(feature = "render")]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        clippy::many_single_char_names,
        clippy::comparison_chain
    )]
    fn render_pixels(&mut self) -> Result<RenderFrame> {
        if self.player.is_empty() {
            return Err(Error::ResetNeeded { method: "render" });
        }

        let (player_sum, _dealer_card, player_usable_ace) = get_obs(&self.player, &self.dealer);

        let card_img_h = SCREEN_H / 3;
        // Original card aspect ratio is 142:197.
        let card_img_w = (card_img_h as f32 * 142.0 / 197.0) as u32;
        let spacing = SCREEN_H / 20;

        // Lazily init sprites (only the card back needs caching; faces decoded on demand).
        let _ = self
            .sprites
            .get_or_insert_with(|| BlackjackSprites::new(card_img_w, card_img_h));

        let canvas = self
            .canvas
            .get_or_insert_with(|| Canvas::new(SCREEN_W, SCREEN_H));

        // Green felt background (same as Gymnasium: rgb(7, 99, 36)).
        canvas.clear(tiny_skia::Color::from_rgba8(7, 99, 36, 255));
        let white = tiny_skia::Color::WHITE;

        // --- Dealer section ---
        // "Dealer: X" label area (white bar approximation since we can't render text).
        let dealer_bar_y = spacing;
        canvas.fill_rect(
            spacing as f32,
            dealer_bar_y as f32,
            160.0,
            20.0,
            tiny_skia::Color::from_rgba8(255, 255, 255, 60),
        );

        // Dealer showing card.
        let dealer_card_img = BlackjackSprites::decode_card(
            self.dealer_top_card_suit,
            self.dealer_top_card_value,
            card_img_w,
            card_img_h,
        );
        let dealer_card_x = (SCREEN_W / 2 - card_img_w - spacing / 2) as i32;
        let dealer_card_y = (dealer_bar_y + 20 + spacing) as i32;
        canvas.blit(dealer_card_x, dealer_card_y, &dealer_card_img);

        // Hidden card (card back).
        let sprites = self.sprites.as_ref().expect("sprites initialized above");
        let hidden_x = (SCREEN_W / 2 + spacing / 2) as i32;
        canvas.blit(hidden_x, dealer_card_y, &sprites.card_back);

        // --- Player section ---
        let player_label_y = dealer_card_y + card_img_h as i32 + (1.5 * spacing as f32) as i32;
        canvas.fill_rect(
            spacing as f32,
            player_label_y as f32,
            100.0,
            20.0,
            tiny_skia::Color::from_rgba8(255, 255, 255, 60),
        );

        // Player sum indicator (large bar showing value proportion to 21).
        let sum_bar_y = player_label_y + 20 + spacing as i32;
        let bar_max_w = (SCREEN_W - 2 * spacing) as f32;
        let bar_w = (player_sum as f32 / 21.0).min(1.0) * bar_max_w;
        let bar_color = if player_sum > 21 {
            tiny_skia::Color::from_rgba8(220, 50, 50, 255) // Bust = red.
        } else if player_sum == 21 {
            tiny_skia::Color::from_rgba8(255, 215, 0, 255) // Blackjack = gold.
        } else {
            tiny_skia::Color::from_rgba8(100, 200, 100, 255) // Normal = green.
        };
        canvas.fill_rect(spacing as f32, sum_bar_y as f32, bar_w, 40.0, bar_color);
        // Border.
        canvas.stroke_line(
            spacing as f32,
            sum_bar_y as f32,
            (SCREEN_W - spacing) as f32,
            sum_bar_y as f32,
            1.0,
            white,
        );
        canvas.stroke_line(
            spacing as f32,
            (sum_bar_y + 40) as f32,
            (SCREEN_W - spacing) as f32,
            (sum_bar_y + 40) as f32,
            1.0,
            white,
        );
        // 21 reference line.
        let ref_x = (SCREEN_W - spacing) as f32;
        canvas.stroke_line(
            ref_x,
            (sum_bar_y - 5) as f32,
            ref_x,
            (sum_bar_y + 45) as f32,
            2.0,
            tiny_skia::Color::from_rgba8(255, 215, 0, 200),
        );

        // Usable ace indicator (purple circle).
        if player_usable_ace == 1 {
            let ace_x = (SCREEN_W / 2) as f32;
            let ace_y = (sum_bar_y + 60) as f32;
            canvas.fill_circle(
                ace_x,
                ace_y,
                10.0,
                tiny_skia::Color::from_rgba8(200, 0, 200, 255),
            );
        }

        match self.render_mode {
            RenderMode::Human => {
                let window = self.window.get_or_insert_with(|| {
                    RenderWindow::new(
                        "Blackjack \u{2014} gmgn",
                        SCREEN_W as usize,
                        SCREEN_H as usize,
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
                    width: SCREEN_W,
                    height: SCREEN_H,
                    data: rgb,
                })
            }
            _ => Ok(RenderFrame::None),
        }
    }

    /// Pick a random card suit for rendering.
    fn random_suit(rng: &mut Rng) -> u8 {
        [b'C', b'D', b'H', b'S'][rng.random_range(0..4)]
    }

    /// Map a dealer card value (1-10) to a display character for PNG lookup.
    fn card_value_char(value: i64, rng: &mut Rng) -> u8 {
        match value {
            1 => b'A',
            2 => b'2',
            3 => b'3',
            4 => b'4',
            5 => b'5',
            6 => b'6',
            7 => b'7',
            8 => b'8',
            9 => b'9',
            10 => [b'T', b'J', b'Q', b'K'][rng.random_range(0..4)],
            _ => b'T',
        }
    }
}

impl Env for BlackjackEnv {
    type Obs = BlackjackObs;
    type Act = i64;
    type ObsSpace = BlackjackObsSpace;
    type ActSpace = Discrete;

    fn step(&mut self, action: &i64) -> Result<StepResult<BlackjackObs>> {
        if self.player.is_empty() {
            return Err(Error::ResetNeeded { method: "step" });
        }
        if !self.action_space.contains(action) {
            return Err(Error::InvalidAction {
                reason: format!("action {action} not in {{0, 1}}"),
            });
        }

        let (terminated, reward) = if *action == 1 {
            // Hit: draw a card.
            self.player.push(draw_card(&mut self.rng));
            if is_bust(&self.player) {
                (true, -1.0)
            } else {
                (false, 0.0)
            }
        } else {
            // Stick: dealer plays, then compare.
            while sum_hand(&self.dealer) < 17 {
                self.dealer.push(draw_card(&mut self.rng));
            }
            let mut reward = cmp_scores(score(&self.player), score(&self.dealer));

            if self.sab && is_natural(&self.player) && !is_natural(&self.dealer) {
                reward = 1.0;
            } else if !self.sab
                && self.natural
                && is_natural(&self.player)
                && (reward - 1.0).abs() < f64::EPSILON
            {
                reward = 1.5;
            }
            (true, reward)
        };

        let obs = get_obs(&self.player, &self.dealer);

        let mut info = HashMap::new();
        info.insert(
            "player".to_owned(),
            InfoValue::String(format!("{:?}", self.player)),
        );
        info.insert(
            "dealer".to_owned(),
            InfoValue::String(format!("{:?}", self.dealer)),
        );

        Ok(StepResult {
            obs,
            reward,
            terminated,
            truncated: false,
            info,
        })
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<BlackjackObs>> {
        if let Some(s) = seed {
            self.rng = rng::create_rng(Some(s));
        }

        self.dealer = draw_hand(&mut self.rng);
        self.player = draw_hand(&mut self.rng);

        // Track dealer's showing card suit/value for PNG rendering.
        let dealer_card_value = self.dealer[0];
        self.dealer_top_card_suit = Self::random_suit(&mut self.rng);
        self.dealer_top_card_value = Self::card_value_char(dealer_card_value, &mut self.rng);

        let obs = get_obs(&self.player, &self.dealer);

        let mut info = HashMap::new();
        info.insert(
            "player".to_owned(),
            InfoValue::String(format!("{:?}", self.player)),
        );
        info.insert(
            "dealer".to_owned(),
            InfoValue::String(format!("{:?}", self.dealer)),
        );

        Ok(ResetResult { obs, info })
    }

    fn render(&mut self) -> Result<RenderFrame> {
        match self.render_mode {
            RenderMode::None => Ok(RenderFrame::None),
            RenderMode::Ansi => {
                if self.player.is_empty() {
                    return Err(Error::ResetNeeded { method: "render" });
                }
                let mut lines = Vec::new();
                lines.push(format!(
                    "Player: {:?} (sum={}, usable_ace={})",
                    self.player,
                    sum_hand(&self.player),
                    usable_ace(&self.player)
                ));
                lines.push(format!(
                    "Dealer: {:?} (showing={})",
                    self.dealer, self.dealer[0]
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

    fn observation_space(&self) -> &BlackjackObsSpace {
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

    fn make_env() -> BlackjackEnv {
        BlackjackEnv::new(BlackjackConfig::default())
    }

    #[test]
    fn reset_produces_valid_observation() {
        let mut env = make_env();
        let r = env.reset(Some(42)).expect("reset");
        assert!(env.observation_space().contains(&r.obs));
        // player_sum should be >= 2 (two cards minimum).
        assert!(r.obs.0 >= 2);
        // dealer_card should be 1..=10.
        assert!((1..=10).contains(&r.obs.1));
        // usable_ace should be 0 or 1.
        assert!(r.obs.2 == 0 || r.obs.2 == 1);
    }

    #[test]
    fn step_without_reset_errors() {
        let mut env = make_env();
        assert!(env.step(&0).is_err());
    }

    #[test]
    fn step_invalid_action_errors() {
        let mut env = make_env();
        env.reset(Some(0)).expect("reset");
        assert!(env.step(&2).is_err());
    }

    #[test]
    fn stick_always_terminates() {
        let mut env = make_env();
        env.reset(Some(42)).expect("reset");
        let r = env.step(&0).expect("stick");
        assert!(r.terminated);
    }

    #[test]
    fn bust_terminates_with_negative_reward() {
        let mut env = make_env();
        env.reset(Some(0)).expect("reset");
        // Force a hand that will bust: player has 20, hit → almost certainly busts.
        env.player = vec![10, 10];
        env.dealer = vec![5, 5];
        let r = env.step(&1).expect("hit");
        if r.terminated {
            assert!((r.reward - (-1.0)).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn natural_blackjack_sab_wins() {
        let mut env = BlackjackEnv::new(BlackjackConfig {
            sab: true,
            ..BlackjackConfig::default()
        });
        env.reset(Some(0)).expect("reset");
        // Force natural: Ace + 10.
        env.player = vec![1, 10];
        env.dealer = vec![5, 5];
        let r = env.step(&0).expect("stick");
        assert!(r.terminated);
        assert!(r.reward >= 1.0, "natural should win, got {}", r.reward);
    }

    #[test]
    fn deterministic_with_seed() {
        let mut e1 = make_env();
        let mut e2 = make_env();

        let r1 = e1.reset(Some(99)).expect("reset");
        let r2 = e2.reset(Some(99)).expect("reset");
        assert_eq!(r1.obs, r2.obs);

        let s1 = e1.step(&1).expect("hit");
        let s2 = e2.step(&1).expect("hit");
        assert_eq!(s1.obs, s2.obs);
        assert!((s1.reward - s2.reward).abs() < f64::EPSILON);
    }

    #[test]
    fn observation_space_contains_checks() {
        let env = make_env();
        let space = env.observation_space();
        // Valid observations.
        assert!(space.contains(&(4, 1, 0)));
        assert!(space.contains(&(21, 10, 1)));
        // Invalid.
        assert!(!space.contains(&(32, 1, 0)));
        assert!(!space.contains(&(4, 11, 0)));
        assert!(!space.contains(&(4, 1, 2)));
    }

    #[test]
    fn card_helpers_correctness() {
        // Hand with usable ace.
        assert!(usable_ace(&[1, 5]));
        assert_eq!(sum_hand(&[1, 5]), 16);
        // No usable ace (would bust).
        assert!(!usable_ace(&[1, 10, 5]));
        assert_eq!(sum_hand(&[1, 10, 5]), 16);
        // Natural.
        assert!(is_natural(&[1, 10]));
        assert!(!is_natural(&[1, 5]));
        assert!(!is_natural(&[10, 5, 6]));
        // Bust.
        assert!(is_bust(&[10, 10, 5]));
        assert!(!is_bust(&[10, 10]));
    }
}
