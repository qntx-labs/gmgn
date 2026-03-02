//! Environment wrappers for modular behavior transformation.
//!
//! Wrappers modify an environment's behavior without altering its underlying
//! code. They can transform observations, actions, rewards, or add bookkeeping
//! such as time limits and episode statistics.
//!
//! Mirrors [Gymnasium wrappers](https://gymnasium.farama.org/api/wrappers/)
//! adapted to Rust's generic type system for zero-cost composition.

mod autoreset;
mod clip_action;
mod clip_reward;
mod filter_observation;
mod flatten_observation;
mod frame_stack;
mod max_and_skip;
mod normalize_observation;
mod normalize_reward;
mod order_enforcing;
mod record_episode_statistics;
mod rescale_action;
mod rescale_observation;
mod sticky_action;
mod time_aware;
mod time_limit;
mod transform_action;
mod transform_observation;
mod transform_reward;

pub use autoreset::Autoreset;
pub use clip_action::ClipAction;
pub use clip_reward::ClipReward;
pub use filter_observation::FilterObservation;
pub use flatten_observation::FlattenObservation;
pub use frame_stack::FrameStackObservation;
pub use max_and_skip::MaxAndSkipObservation;
pub use normalize_observation::NormalizeObservation;
pub use normalize_reward::NormalizeReward;
pub use order_enforcing::OrderEnforcing;
pub use record_episode_statistics::RecordEpisodeStatistics;
pub use rescale_action::RescaleAction;
pub use rescale_observation::RescaleObservation;
pub use sticky_action::StickyAction;
pub use time_aware::TimeAwareObservation;
pub use time_limit::TimeLimit;
pub use transform_action::TransformAction;
pub use transform_observation::TransformObservation;
pub use transform_reward::TransformReward;
