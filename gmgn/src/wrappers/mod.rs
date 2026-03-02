//! Environment wrappers for modular behavior transformation.
//!
//! Wrappers modify an environment's behavior without altering its underlying
//! code. They can transform observations, actions, rewards, or add bookkeeping
//! such as time limits and episode statistics.
//!
//! Mirrors [Gymnasium wrappers](https://gymnasium.farama.org/api/wrappers/)
//! adapted to Rust's generic type system for zero-cost composition.

mod clip_action;
mod normalize_observation;
mod normalize_reward;
mod record_episode_statistics;
mod time_limit;

pub use clip_action::ClipAction;
pub use normalize_observation::NormalizeObservation;
pub use normalize_reward::NormalizeReward;
pub use record_episode_statistics::RecordEpisodeStatistics;
pub use time_limit::TimeLimit;
