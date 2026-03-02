//! `LunarLander` discrete demo — random agent with human rendering.
//!
//! Run: `cargo run --example lunar_lander --features box2d,render`

#![allow(clippy::print_stdout, clippy::unwrap_used)]

use gmgn::env::{Env, RenderMode};
use gmgn::envs::box2d::{LunarLanderConfig, LunarLanderEnv};
use gmgn::space::Space;

fn main() {
    let mut env = LunarLanderEnv::new(LunarLanderConfig {
        render_mode: RenderMode::Human,
        ..LunarLanderConfig::default()
    })
    .unwrap();

    let mut rng = gmgn::rng::create_rng(Some(42));

    for ep in 1..=5 {
        env.reset(None).unwrap();
        let mut total = 0.0;

        for _ in 0..1000 {
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
