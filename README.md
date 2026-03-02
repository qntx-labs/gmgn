# gmgn

[![CI][ci-badge]][ci-url]
[![crates.io][crate-badge]][crate-url]
[![docs.rs][doc-badge]][doc-url]
[![License][license-badge]][license-url]
[![Rust][rust-badge]][rust-url]

[ci-badge]: https://github.com/qntx/gmgn/actions/workflows/rust.yml/badge.svg
[ci-url]: https://github.com/qntx/gmgn/actions/workflows/rust.yml
[crate-badge]: https://img.shields.io/crates/v/gmgn.svg
[crate-url]: https://crates.io/crates/gmgn
[doc-badge]: https://img.shields.io/docsrs/gmgn.svg
[doc-url]: https://docs.rs/gmgn
[license-badge]: https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg
[license-url]: LICENSE-MIT
[rust-badge]: https://img.shields.io/badge/rust-edition%202024-orange.svg
[rust-url]: https://doc.rust-lang.org/edition-guide/

**Pure-Rust reinforcement learning environments library — a faithful port of [Gymnasium](https://gymnasium.farama.org/) with zero-cost abstractions, native rendering, and no unsafe code in user-facing APIs.**

gmgn provides a generic `Env` trait with associated types for compile-time monomorphization, a dynamic `DynEnv` trait with a global `make("CartPole-v1")` registry, 14 built-in environments across three categories (classic control, toy text, Box2D), 21 composable wrappers, vectorized execution, and real-time graphical rendering via `tiny-skia` + `minifb` — all without leaving Rust.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
gmgn = "0.4"
```

### Static API (generic, zero-cost)

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

### Dynamic API (registry)

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

### With Rendering

```rust
use gmgn::env::{Env, RenderMode};
use gmgn::envs::classic_control::{CartPoleEnv, CartPoleConfig};
use gmgn::space::Space;

let mut env = CartPoleEnv::new(CartPoleConfig {
    render_mode: RenderMode::Human,
    ..CartPoleConfig::default()
}).unwrap();

let mut rng = gmgn::rng::create_rng(Some(42));
env.reset(None).unwrap();

for _ in 0..500 {
    let action = env.action_space().sample(&mut rng);
    let s = env.step(&action).unwrap();
    env.render().unwrap(); // opens a native OS window at 50 FPS
    if s.terminated { break; }
}
```

## Architecture

gmgn uses a **dual-track** design that mirrors Gymnasium while leveraging Rust's type system:

- **Static track** — The generic `Env` trait uses associated types (`Obs`, `Act`, `ObsSpace`, `ActSpace`) for full monomorphization. No boxing, no vtables, no runtime dispatch. Wrappers compose via generic nesting (`TimeLimit<ClipAction<CartPoleEnv>>`).
- **Dynamic track** — The `DynEnv` trait + global registry enables `make("CartPole-v1")` style creation with type-erased `DynValue` observations and actions. Auto-wraps with `TimeLimit` and `OrderEnforcing` per registered `EnvSpec`.
- **Spaces** — 10 space types (`Discrete`, `BoundedSpace`, `MultiDiscrete`, `MultiBinary`, `Dict`, `Tuple`, `OneOf`, `Sequence`, `Graph`, `Text`) each implement the `Space` trait with `sample()`, `contains()`, `flatdim()`, and `flatten()`.
- **Rendering** — Feature-gated `Canvas` (2D drawing surface over `tiny-skia` Pixmap) + `RenderWindow` (native OS window via `minifb`). Supports `RenderMode::Human` (live window) and `RenderMode::RgbArray` (pixel buffer).
- **Vectorized** — `SyncVectorEnv` and `AsyncVectorEnv` with configurable `AutoresetMode` (`NextStep` / `SameStep` / `Disabled`), plus `VecRecordEpisodeStatistics` wrapper.

## Environments

### Classic Control

| Id | Observation | Action | Max Steps | Reward Threshold |
| --- | --- | --- | --- | --- |
| `CartPole-v1` | `Vec<f32>` (4) | `i64` · Discrete(2) | 500 | 475.0 |
| `MountainCar-v0` | `Vec<f32>` (2) | `i64` · Discrete(3) | 200 | −110.0 |
| `MountainCarContinuous-v0` | `Vec<f32>` (2) | `Vec<f32>` (1) | 999 | 90.0 |
| `Pendulum-v1` | `Vec<f32>` (3) | `Vec<f32>` (1) | 200 | — |
| `Acrobot-v1` | `Vec<f32>` (6) | `i64` · Discrete(3) | 500 | −100.0 |

### Toy Text

| Id | Observation | Action | Max Steps | Reward Threshold |
| --- | --- | --- | --- | --- |
| `FrozenLake-v1` | `i64` | `i64` · Discrete(4) | 100 | 0.70 |
| `FrozenLake8x8-v1` | `i64` | `i64` · Discrete(4) | 200 | 0.85 |
| `Taxi-v3` | `i64` | `i64` · Discrete(6) | 200 | 8.0 |
| `CliffWalking-v1` | `i64` | `i64` · Discrete(4) | — | — |
| `Blackjack-v1` | `(i64, i64, i64)` | `i64` · Discrete(2) | — | — |

### Box2D (feature: `box2d`)

| Id | Observation | Action | Max Steps | Reward Threshold |
| --- | --- | --- | --- | --- |
| `LunarLander-v3` | `Vec<f32>` (8) | `i64` · Discrete(4) | 1000 | 200.0 |
| `BipedalWalker-v3` | `Vec<f32>` (24) | `Vec<f32>` (4) | 1600 | 300.0 |
| `BipedalWalkerHardcore-v3` | `Vec<f32>` (24) | `Vec<f32>` (4) | 2000 | 300.0 |

All environments are rigorously validated against the official Gymnasium Python implementations for observation spaces, reward shaping, physics parameters, and termination conditions.

## Wrappers

21 composable wrappers mirroring [Gymnasium wrappers](https://gymnasium.farama.org/api/wrappers/):

| Wrapper | Category | Description |
| --- | --- | --- |
| `TimeLimit` | Utility | Truncate episodes after N steps |
| `OrderEnforcing` | Utility | Enforce `reset()` before `step()` |
| `Autoreset` | Utility | Auto-reset on episode end |
| `RecordEpisodeStatistics` | Utility | Track per-episode reward, length, time |
| `ClipAction` | Action | Clip continuous actions to space bounds |
| `RescaleAction` | Action | Affine-map actions to a new range |
| `TransformAction` | Action | Apply arbitrary action transform |
| `DiscretizeAction` | Action | Map discrete indices to continuous bins |
| `StickyAction` | Action | Repeat previous action with probability p |
| `ClipReward` | Reward | Clip rewards to a range |
| `TransformReward` | Reward | Apply arbitrary reward transform |
| `NormalizeReward` | Reward | Running-mean normalize rewards |
| `NormalizeObservation` | Observation | Running-mean normalize observations |
| `RescaleObservation` | Observation | Affine-map observations to a new range |
| `TransformObservation` | Observation | Apply arbitrary observation transform |
| `FilterObservation` | Observation | Select a subset of observation dimensions |
| `FlattenObservation` | Observation | Flatten structured observations |
| `FrameStackObservation` | Observation | Stack N consecutive frames |
| `DelayObservation` | Observation | Delay observation by N steps |
| `MaxAndSkipObservation` | Observation | Max-pool over N skipped frames |
| `TimeAwareObservation` | Observation | Append normalized time step to obs |

## Feature Flags

| Feature | Default | Description |
| --- | --- | --- |
| `render` | Yes | Real-time graphical rendering via `tiny-skia` + `minifb` |
| `box2d` | No | Box2D physics environments (`LunarLander`, `BipedalWalker`) via `box2d-rs` |

```bash
# Default (classic control + toy text + rendering)
cargo add gmgn

# With Box2D environments
cargo add gmgn --features box2d

# Minimal (no rendering, no Box2D)
cargo add gmgn --no-default-features
```

## Examples

```bash
# Classic control with rendering
cargo run --example cartpole
cargo run --example mountain_car
cargo run --example continuous_mountain_car
cargo run --example pendulum
cargo run --example acrobot

# Toy text
cargo run --example frozen_lake
cargo run --example taxi
cargo run --example cliff_walking
cargo run --example blackjack

# Box2D (requires feature flag)
cargo run --example lunar_lander --features box2d
cargo run --example bipedal_walker --features box2d
```

## Gymnasium Compatibility

gmgn aims to be a **faithful Rust port** of [Gymnasium](https://gymnasium.farama.org/) (Farama Foundation). Every environment is implemented by line-by-line comparison with the official Python source, including:

- Identical physics parameters, constants, and coordinate systems
- Matching observation and action space definitions
- Equivalent reward shaping formulas and termination conditions
- Aligned RNG consumption order for seed-reproducible trajectories
- Pixel-accurate rendering matching Gymnasium's `pygame` output

### Intentional Differences

The following divergences are by design, leveraging Rust's strengths:

| Gymnasium (Python) | gmgn (Rust) | Rationale |
| --- | --- | --- |
| `gym.Env` base class | `Env` trait with associated types | Zero-cost generics, no vtable overhead |
| `Box` / `Discrete` spaces | `BoundedSpace` / `Discrete` | Naming clarity (`Box` is reserved in Rust) |
| `np.random` seed management | `rand_pcg::Pcg64Mcg` via `create_rng(seed)` | Deterministic, portable PRNG |
| `mask` / `probability` sampling | Not implemented | Users can sample directly with `rand` |
| `to_jsonable` / `from_jsonable` | Not implemented | Use `serde` ecosystem instead |
| `dtype` parameter | Compile-time types | Rust's type system replaces runtime dtype |
| `gymnasium.Wrapper` base class | Generic nesting (`W<E: Env>`) | Zero-cost composition, no inheritance |

## Minimum Supported Rust Version

Rust **edition 2024** (nightly or stable 1.85+).

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project shall be dual-licensed as above, without any additional terms or conditions.
