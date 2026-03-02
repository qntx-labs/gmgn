//! Pendulum with real-time rendering.
//!
//! Run: `cargo run --example pendulum --features render`

#![allow(clippy::print_stdout, clippy::unwrap_used)]

use gmgn::env::{Env, RenderMode};
use gmgn::envs::classic_control::{PendulumConfig, PendulumEnv};
use gmgn::space::Space;

fn main() {
    let mut env = PendulumEnv::new(PendulumConfig {
        render_mode: RenderMode::Human,
        ..PendulumConfig::default()
    })
    .unwrap();

    let mut rng = gmgn::rng::create_rng(Some(42));

    for ep in 1..=3 {
        env.reset(None).unwrap();
        let mut total = 0.0;

        for _ in 0..200 {
            let action = env.action_space().sample(&mut rng);
            let s = env.step(&action).unwrap();
            env.render().unwrap();
            total += s.reward;
        }
        println!("Episode {ep}: reward = {total:.1}");
    }
}
