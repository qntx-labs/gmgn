//! Native OS window for real-time environment display.

use minifb::{Window, WindowOptions};

use crate::error::{Error, Result};
use crate::render::Canvas;

/// A native window that displays [`Canvas`] frames at a target FPS.
///
/// Wraps `minifb::Window` and handles frame timing, event pumping,
/// and pixel format conversion automatically.
pub struct RenderWindow {
    window: Window,
    width: usize,
    height: usize,
    target_fps: usize,
}

impl std::fmt::Debug for RenderWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderWindow")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("target_fps", &self.target_fps)
            .finish_non_exhaustive()
    }
}

impl RenderWindow {
    /// Create a new render window with the given title and dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the window cannot be created.
    pub fn new(title: &str, width: usize, height: usize, target_fps: usize) -> Result<Self> {
        let mut window = Window::new(
            title,
            width,
            height,
            WindowOptions {
                resize: false,
                scale_mode: minifb::ScaleMode::AspectRatioStretch,
                ..WindowOptions::default()
            },
        )
        .map_err(|e| Error::MissingDependency(format!("failed to create window: {e}")))?;

        // Set target frame rate (minifb handles timing internally).
        if target_fps > 0 {
            window.set_target_fps(target_fps);
        }

        Ok(Self {
            window,
            width,
            height,
            target_fps,
        })
    }

    /// Display a canvas frame in the window.
    ///
    /// Converts the canvas pixel buffer and updates the window. This call
    /// blocks until the next frame deadline based on the target FPS.
    ///
    /// # Errors
    ///
    /// Returns an error if the window update fails.
    pub fn show(&mut self, canvas: &Canvas) -> Result<()> {
        let buffer = canvas.pixels_argb32();
        self.window
            .update_with_buffer(&buffer, self.width, self.height)
            .map_err(|e| Error::MissingDependency(format!("window update failed: {e}")))?;
        Ok(())
    }

    /// Check if the window is still open (user hasn't closed it).
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.window.is_open()
    }

    /// The target frames per second.
    #[must_use]
    pub const fn target_fps(&self) -> usize {
        self.target_fps
    }
}
