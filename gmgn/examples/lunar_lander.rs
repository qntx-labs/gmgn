//! `LunarLander` discrete demo — random agent.
//!
//! Run: `cargo run --example lunar_lander --features box2d`

#![allow(clippy::print_stdout, clippy::unwrap_used)]

use gmgn::env::Env;
use gmgn::envs::box2d::{LunarLanderConfig, LunarLanderEnv};
use gmgn::space::Space;

fn main() {
    let mut env = LunarLanderEnv::new(LunarLanderConfig::default()).unwrap();

    let mut rng = gmgn::rng::create_rng(Some(42));

    for ep in 1..=5 {
        env.reset(None).unwrap();
        let mut total = 0.0;

        for _ in 0..1000 {
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
