//! `CartPole` with real-time rendering.
//!
//! Run: `cargo run --example cartpole --features render`

#![allow(clippy::print_stdout, clippy::unwrap_used)]

use gmgn::env::{Env, RenderMode};
use gmgn::envs::classic_control::{CartPoleConfig, CartPoleEnv};
use gmgn::space::Space;

fn main() {
    let mut env = CartPoleEnv::new(CartPoleConfig {
        render_mode: RenderMode::Human,
        ..CartPoleConfig::default()
    })
    .unwrap();

    let mut rng = gmgn::rng::create_rng(Some(42));

    for ep in 1..=5 {
        env.reset(None).unwrap();
        let mut total = 0.0;

        for _ in 0..500 {
            let action = env.action_space().sample(&mut rng);
            let s = env.step(&action).unwrap();
            env.render().unwrap();
            total += s.reward;
            if s.terminated {
                break;
            }
        }
        println!("Episode {ep}: reward = {total:.0}");
    }
}
