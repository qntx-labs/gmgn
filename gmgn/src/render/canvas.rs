//! Software-rendered 2D canvas built on `tiny-skia`.

use tiny_skia::{Color, FillRule, LineCap, Paint, PathBuilder, Pixmap, Stroke, Transform};

/// A 2D drawing surface backed by a `tiny-skia` [`Pixmap`].
///
/// Provides high-level drawing primitives (rectangles, circles, lines,
/// polygons) with anti-aliasing, matching the visual quality of Gymnasium's
/// pygame/gfxdraw rendering.
#[derive(Debug)]
pub struct Canvas {
    pixmap: Pixmap,
}

impl Canvas {
    /// Create a new canvas with the given dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `width` or `height` is zero.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            pixmap: Pixmap::new(width, height).expect("non-zero canvas dimensions"),
        }
    }

    /// Width of the canvas in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.pixmap.width()
    }

    /// Height of the canvas in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.pixmap.height()
    }

    /// Fill the entire canvas with a solid color.
    pub fn clear(&mut self, color: Color) {
        self.pixmap.fill(color);
    }

    /// Draw a filled rectangle.
    pub fn fill_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color) {
        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;

        if let Some(rect) = tiny_skia::Rect::from_xywh(x, y, w, h) {
            let path = PathBuilder::from_rect(rect);
            self.pixmap.fill_path(
                &path,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }
    }

    /// Draw a filled circle.
    pub fn fill_circle(&mut self, cx: f32, cy: f32, radius: f32, color: Color) {
        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;

        if let Some(path) = {
            let mut pb = PathBuilder::new();
            pb.push_circle(cx, cy, radius);
            pb.finish()
        } {
            self.pixmap.fill_path(
                &path,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }
    }

    /// Draw a horizontal line spanning the full width at the given y coordinate.
    pub fn hline(&mut self, y: f32, thickness: f32, color: Color) {
        let w = self.pixmap.width() as f32;
        self.stroke_line(0.0, y, w, y, thickness, color);
    }

    /// Draw a line between two points with a given thickness.
    pub fn stroke_line(
        &mut self,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        thickness: f32,
        color: Color,
    ) {
        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;

        let stroke = Stroke {
            width: thickness,
            line_cap: LineCap::Round,
            ..Stroke::default()
        };

        if let Some(path) = {
            let mut pb = PathBuilder::new();
            pb.move_to(x1, y1);
            pb.line_to(x2, y2);
            pb.finish()
        } {
            self.pixmap
                .stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        }
    }

    /// Draw a filled polygon from a list of `(x, y)` vertices.
    pub fn fill_polygon(&mut self, vertices: &[(f32, f32)], color: Color) {
        if vertices.len() < 3 {
            return;
        }

        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;

        if let Some(path) = {
            let mut pb = PathBuilder::new();
            pb.move_to(vertices[0].0, vertices[0].1);
            for &(x, y) in &vertices[1..] {
                pb.line_to(x, y);
            }
            pb.close();
            pb.finish()
        } {
            self.pixmap.fill_path(
                &path,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }
    }

    /// Draw an anti-aliased polygon outline.
    pub fn stroke_polygon(&mut self, vertices: &[(f32, f32)], thickness: f32, color: Color) {
        if vertices.len() < 3 {
            return;
        }

        let mut paint = Paint::default();
        paint.set_color(color);
        paint.anti_alias = true;

        let stroke = Stroke {
            width: thickness,
            ..Stroke::default()
        };

        if let Some(path) = {
            let mut pb = PathBuilder::new();
            pb.move_to(vertices[0].0, vertices[0].1);
            for &(x, y) in &vertices[1..] {
                pb.line_to(x, y);
            }
            pb.close();
            pb.finish()
        } {
            self.pixmap
                .stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        }
    }

    /// Get the raw pixel data as an RGBA byte slice.
    #[must_use]
    pub fn pixels_rgba(&self) -> &[u8] {
        self.pixmap.data()
    }

    /// Convert the pixel buffer to a `Vec<u32>` in `0xAARRGGBB` format,
    /// suitable for `minifb::Window::update_with_buffer`.
    #[must_use]
    pub fn pixels_argb32(&self) -> Vec<u32> {
        self.pixmap
            .pixels()
            .iter()
            .map(|px| {
                let r = u32::from(px.red());
                let g = u32::from(px.green());
                let b = u32::from(px.blue());
                let a = u32::from(px.alpha());
                (a << 24) | (r << 16) | (g << 8) | b
            })
            .collect()
    }

    /// Convert the pixel buffer to an RGB `Vec<u8>` (3 bytes per pixel),
    /// suitable for [`RenderFrame::RgbArray`](crate::env::RenderFrame::RgbArray).
    #[must_use]
    pub fn pixels_rgb(&self) -> Vec<u8> {
        self.pixmap
            .pixels()
            .iter()
            .flat_map(|px| [px.red(), px.green(), px.blue()])
            .collect()
    }
}
