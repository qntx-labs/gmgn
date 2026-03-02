//! `BipedalWalker` discrete demo — random agent.
//!
//! Run: `cargo run --example bipedal_walker --features box2d`

#![allow(clippy::print_stdout, clippy::unwrap_used)]

use gmgn::env::Env;
use gmgn::envs::box2d::{BipedalWalkerConfig, BipedalWalkerEnv};
use gmgn::space::Space;

fn main() {
    let mut env = BipedalWalkerEnv::new(BipedalWalkerConfig::default()).unwrap();

    let mut rng = gmgn::rng::create_rng(Some(42));

    for ep in 1..=5 {
        env.reset(None).unwrap();
        let mut total = 0.0;

        for _ in 0..1600 {
            let action = env.action_space().sample(&mut rng);
            let s = env.step(&action).unwrap();
            total += s.reward;
            if s.terminated {
                break;
            }
        }
        println!("Episode {ep}: reward = {total:.1}");
    }
}
