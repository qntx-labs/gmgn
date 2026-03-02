//! `MountainCarContinuous` with real-time rendering.
//!
//! Run: `cargo run --example continuous_mountain_car --features render`

#![allow(clippy::print_stdout, clippy::unwrap_used)]

use gmgn::env::{Env, RenderMode};
use gmgn::envs::classic_control::{ContinuousMountainCarConfig, ContinuousMountainCarEnv};
use gmgn::space::Space;

fn main() {
    let mut env = ContinuousMountainCarEnv::new(ContinuousMountainCarConfig {
        render_mode: RenderMode::Human,
        ..ContinuousMountainCarConfig::default()
    })
    .unwrap();

    let mut rng = gmgn::rng::create_rng(Some(42));

    for ep in 1..=3 {
        env.reset(None).unwrap();
        let mut total = 0.0;

        for _ in 0..999 {
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
