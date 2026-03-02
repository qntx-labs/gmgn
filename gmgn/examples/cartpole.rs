//! CartPole interactive demo with real-time window rendering.
//!
//! Run with: `cargo run --example cartpole`

use gmgn::env::{Env, RenderMode};
use gmgn::envs::classic_control::{CartPoleConfig, CartPoleEnv};
use gmgn::rng;
use gmgn::space::Space;

fn main() {
    let mut env = CartPoleEnv::new(CartPoleConfig {
        render_mode: RenderMode::Human,
        ..CartPoleConfig::default()
    })
    .expect("failed to create CartPole environment");

    let mut rng = rng::create_rng(Some(42));
    let mut episode = 0_u32;
    let mut total_reward = 0.0_f64;
    let mut step_count = 0_u32;

    println!("CartPole — gmgn demo");
    println!("Close the window to exit.\n");

    env.reset(Some(0)).expect("failed to reset");

    loop {
        // Sample a random action (0 = left, 1 = right).
        let action = env.action_space().sample(&mut rng);

        let result = env.step(&action).expect("step failed");
        total_reward += result.reward;
        step_count += 1;

        // Render to window.
        env.render().expect("render failed");

        if result.terminated || step_count >= 500 {
            episode += 1;
            println!("Episode {episode}: steps = {step_count}, reward = {total_reward:.0}",);

            // Reset for next episode.
            total_reward = 0.0;
            step_count = 0;
            env.reset(None).expect("failed to reset");

            // Check if window was closed.
            #[cfg(feature = "render")]
            {
                // The render call above handles window-close detection.
                // After a few episodes, a closed window returns RenderFrame::None.
            }
        }
    }
}
