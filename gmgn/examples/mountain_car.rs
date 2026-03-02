//! `MountainCar` with real-time rendering.
//!
//! Run: `cargo run --example mountain_car --features render`

#![allow(clippy::print_stdout, clippy::unwrap_used)]

use gmgn::env::{Env, RenderMode};
use gmgn::envs::classic_control::{MountainCarConfig, MountainCarEnv};
use gmgn::space::Space;

fn main() {
    let mut env = MountainCarEnv::new(MountainCarConfig {
        render_mode: RenderMode::Human,
        ..MountainCarConfig::default()
    })
    .unwrap();

    let mut rng = gmgn::rng::create_rng(Some(42));

    for ep in 1..=3 {
        env.reset(None).unwrap();
        let mut steps = 0_u32;

        for _ in 0..200 {
            let action = env.action_space().sample(&mut rng);
            let s = env.step(&action).unwrap();
            env.render().unwrap();
            steps += 1;
            if s.terminated {
                break;
            }
        }
        println!("Episode {ep}: steps = {steps}");
    }
}
