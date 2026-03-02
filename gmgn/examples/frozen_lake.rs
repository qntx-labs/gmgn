//! `FrozenLake` with real-time rendering.
//!
//! Run: `cargo run --example frozen_lake --features render`

#![allow(clippy::print_stdout, clippy::unwrap_used)]

use gmgn::env::{Env, RenderMode};
use gmgn::envs::toy_text::{FrozenLakeConfig, FrozenLakeEnv};
use gmgn::space::Space;

fn main() {
    let mut env = FrozenLakeEnv::new(FrozenLakeConfig {
        render_mode: RenderMode::Human,
        is_slippery: true,
        ..FrozenLakeConfig::default()
    })
    .unwrap();

    let mut rng = gmgn::rng::create_rng(Some(42));

    for ep in 1..=5 {
        env.reset(None).unwrap();
        env.render().unwrap();
        let mut total = 0.0;

        for _ in 0..200 {
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
