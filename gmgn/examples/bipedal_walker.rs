//! `BipedalWalker` continuous demo — random agent with human rendering.
//!
//! Run: `cargo run --example bipedal_walker --features box2d,render`

#![allow(clippy::print_stdout, clippy::unwrap_used)]

use gmgn::env::{Env, RenderMode};
use gmgn::envs::box2d::{BipedalWalkerConfig, BipedalWalkerEnv};
use gmgn::space::Space;

fn main() {
    let mut env = BipedalWalkerEnv::new(BipedalWalkerConfig {
        render_mode: RenderMode::Human,
        ..BipedalWalkerConfig::default()
    })
    .unwrap();

    let mut rng = gmgn::rng::create_rng(Some(42));

    for ep in 1..=5 {
        env.reset(None).unwrap();
        let mut total = 0.0;

        for _ in 0..1600 {
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
