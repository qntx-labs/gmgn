//! `Blackjack` with real-time rendering.
//!
//! Run: `cargo run --example blackjack --features render`

#![allow(clippy::print_stdout, clippy::unwrap_used)]

use gmgn::env::{Env, RenderMode};
use gmgn::envs::toy_text::{BlackjackConfig, BlackjackEnv};
use gmgn::space::Space;

fn main() {
    let mut env = BlackjackEnv::new(BlackjackConfig {
        render_mode: RenderMode::Human,
        ..BlackjackConfig::default()
    });

    let mut rng = gmgn::rng::create_rng(Some(42));

    for ep in 1..=10 {
        env.reset(None).unwrap();
        env.render().unwrap();
        let mut total = 0.0;

        loop {
            let action = env.action_space().sample(&mut rng);
            let s = env.step(&action).unwrap();
            env.render().unwrap();
            total += s.reward;
            if s.terminated {
                break;
            }
        }
        println!("Episode {ep}: reward = {total:.1}");
    }
}
