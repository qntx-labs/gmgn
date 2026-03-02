# gmgn

A pure-Rust reinforcement learning environments library, inspired by [Gymnasium](https://gymnasium.farama.org/).

[![Crates.io](https://img.shields.io/crates/v/gmgn.svg)](https://crates.io/crates/gmgn)
[![License](https://img.shields.io/crates/l/gmgn.svg)](LICENSE-MIT)

## Features

- **Zero-cost generic `Env` trait** — associated types, no boxing, full monomorphization
- **Dynamic `DynEnv` trait + global registry** — `make("CartPole-v1")` just like Gymnasium
- **Spaces** — `Discrete`, `BoundedSpace` (Box), `MultiDiscrete`, `MultiBinary`
- **Classic control environments** — CartPole, MountainCar, Pendulum, Acrobot
- **Wrappers** — `TimeLimit`, `ClipAction`, `RecordEpisodeStatistics`, `NormalizeObservation`, `NormalizeReward`
- **Vectorized environments** — `SyncVectorEnv` with configurable autoreset
- **Rendering** — `tiny-skia` + `minifb` canvas (feature-gated)

## Quick Start

### Static (generic, zero-cost)

```rust
use gmgn::prelude::*;
use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};

let mut env = CartPoleEnv::new(CartPoleConfig::default()).unwrap();
let _reset = env.reset(Some(42)).unwrap();

for _ in 0..200 {
    let action = env.action_space().sample(&mut gmgn::rng::create_rng(None));
    let step = env.step(&action).unwrap();
    if step.terminated {
        break;
    }
}
```

### Dynamic (registry)

```rust
use gmgn::registry::{self, DynValue};

registry::register_builtins();
let mut env = registry::make("CartPole-v1").unwrap();
env.reset_dyn(Some(42)).unwrap();
let step = env.step_dyn(&DynValue::Discrete(1)).unwrap();
```

### Vectorized

```rust
use gmgn::prelude::*;
use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
use gmgn::vector::SyncVectorEnv;

let envs: Vec<_> = (0..4)
    .map(|_| CartPoleEnv::new(CartPoleConfig::default()).unwrap())
    .collect();
let mut vec_env = SyncVectorEnv::new(envs).unwrap();
let reset = vec_env.reset(Some(0)).unwrap();
assert_eq!(reset.obs.len(), 4);
```

## Environments

| Id | Obs | Act | Max Steps |
| ---- | ---- | ---- | ----------- |
| `CartPole-v1` | `Vec<f32>` (4) | `i64` (Discrete 2) | 500 |
| `MountainCar-v0` | `Vec<f32>` (2) | `i64` (Discrete 3) | 200 |
| `Pendulum-v1` | `Vec<f32>` (3) | `Vec<f32>` (1) | 200 |
| `Acrobot-v1` | `Vec<f32>` (6) | `i64` (Discrete 3) | 500 |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
