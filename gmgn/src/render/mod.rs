//! 2D rendering system for environment visualization.
//!
//! Provides a [`Canvas`] abstraction over `tiny-skia` for drawing primitives
//! (rectangles, circles, lines, polygons) and a [`RenderWindow`] for
//! displaying frames in a native OS window via `minifb`.
//!
//! This module is only available when the `render` feature is enabled.

mod canvas;
pub mod sprites;
mod window;

pub use canvas::Canvas;
pub use window::RenderWindow;
