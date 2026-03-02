//! Embedded PNG sprite assets for toy-text environment rendering.
//!
//! All images are compiled into the binary via `include_bytes!` and decoded
//! on first use.  Each environment caches its own set of pre-scaled
//! [`tiny_skia::Pixmap`] sprites so decoding happens only once per session.

use tiny_skia::Pixmap;

use super::Canvas;

/// Raw PNG bytes for `FrozenLake` / `CliffWalking` shared sprites.
pub mod frozen_lake {
    /// Ice tile background.
    pub const ICE: &[u8] = include_bytes!("../../assets/toy_text/ice.png");
    /// Hole (water).
    pub const HOLE: &[u8] = include_bytes!("../../assets/toy_text/hole.png");
    /// Cracked hole (elf fell in).
    pub const CRACKED_HOLE: &[u8] = include_bytes!("../../assets/toy_text/cracked_hole.png");
    /// Goal (gift box).
    pub const GOAL: &[u8] = include_bytes!("../../assets/toy_text/goal.png");
    /// Start position (stool).
    pub const STOOL: &[u8] = include_bytes!("../../assets/toy_text/stool.png");
    /// Elf facing left.
    pub const ELF_LEFT: &[u8] = include_bytes!("../../assets/toy_text/elf_left.png");
    /// Elf facing down.
    pub const ELF_DOWN: &[u8] = include_bytes!("../../assets/toy_text/elf_down.png");
    /// Elf facing right.
    pub const ELF_RIGHT: &[u8] = include_bytes!("../../assets/toy_text/elf_right.png");
    /// Elf facing up.
    pub const ELF_UP: &[u8] = include_bytes!("../../assets/toy_text/elf_up.png");
}

/// Raw PNG bytes for CliffWalking-specific sprites.
pub mod cliff_walking {
    /// Mountain background tile variant 1.
    pub const BG1: &[u8] = include_bytes!("../../assets/toy_text/mountain_bg1.png");
    /// Mountain background tile variant 2.
    pub const BG2: &[u8] = include_bytes!("../../assets/toy_text/mountain_bg2.png");
    /// Cliff tile.
    pub const CLIFF: &[u8] = include_bytes!("../../assets/toy_text/mountain_cliff.png");
    /// Near-cliff tile variant 1.
    pub const NEAR_CLIFF1: &[u8] = include_bytes!("../../assets/toy_text/mountain_near-cliff1.png");
    /// Near-cliff tile variant 2.
    pub const NEAR_CLIFF2: &[u8] = include_bytes!("../../assets/toy_text/mountain_near-cliff2.png");
    /// Goal (cookie).
    pub const COOKIE: &[u8] = include_bytes!("../../assets/toy_text/cookie.png");
}

/// Raw PNG bytes for Taxi environment sprites.
pub mod taxi {
    /// Background tile.
    pub const BACKGROUND: &[u8] = include_bytes!("../../assets/toy_text/taxi_background.png");
    /// Cab facing front (south).
    pub const CAB_FRONT: &[u8] = include_bytes!("../../assets/toy_text/cab_front.png");
    /// Cab facing rear (north).
    pub const CAB_REAR: &[u8] = include_bytes!("../../assets/toy_text/cab_rear.png");
    /// Cab facing right (east).
    pub const CAB_RIGHT: &[u8] = include_bytes!("../../assets/toy_text/cab_right.png");
    /// Cab facing left (west).
    pub const CAB_LEFT: &[u8] = include_bytes!("../../assets/toy_text/cab_left.png");
    /// Passenger sprite.
    pub const PASSENGER: &[u8] = include_bytes!("../../assets/toy_text/passenger.png");
    /// Destination hotel sprite.
    pub const HOTEL: &[u8] = include_bytes!("../../assets/toy_text/hotel.png");
    /// Gridworld median: left cap.
    pub const MEDIAN_LEFT: &[u8] =
        include_bytes!("../../assets/toy_text/gridworld_median_left.png");
    /// Gridworld median: horizontal segment.
    pub const MEDIAN_HORIZ: &[u8] =
        include_bytes!("../../assets/toy_text/gridworld_median_horiz.png");
    /// Gridworld median: right cap.
    pub const MEDIAN_RIGHT: &[u8] =
        include_bytes!("../../assets/toy_text/gridworld_median_right.png");
    /// Gridworld median: top cap.
    pub const MEDIAN_TOP: &[u8] = include_bytes!("../../assets/toy_text/gridworld_median_top.png");
    /// Gridworld median: vertical segment.
    pub const MEDIAN_VERT: &[u8] =
        include_bytes!("../../assets/toy_text/gridworld_median_vert.png");
    /// Gridworld median: bottom cap.
    pub const MEDIAN_BOTTOM: &[u8] =
        include_bytes!("../../assets/toy_text/gridworld_median_bottom.png");
}

/// Raw PNG bytes for Blackjack card images.
pub mod blackjack {
    /// Card back.
    pub const CARD_BACK: &[u8] = include_bytes!("../../assets/toy_text/Card.png");

    /// Get the PNG bytes for a specific card face.
    ///
    /// `suit` is one of `'C'`, `'D'`, `'H'`, `'S'`.
    /// `value` is one of `"2"`..`"9"`, `"T"`, `"J"`, `"Q"`, `"K"`, `"A"`.
    #[must_use]
    pub const fn card_png(suit: u8, value: u8) -> &'static [u8] {
        match (suit, value) {
            (b'C', b'2') => include_bytes!("../../assets/toy_text/C2.png"),
            (b'C', b'3') => include_bytes!("../../assets/toy_text/C3.png"),
            (b'C', b'4') => include_bytes!("../../assets/toy_text/C4.png"),
            (b'C', b'5') => include_bytes!("../../assets/toy_text/C5.png"),
            (b'C', b'6') => include_bytes!("../../assets/toy_text/C6.png"),
            (b'C', b'7') => include_bytes!("../../assets/toy_text/C7.png"),
            (b'C', b'8') => include_bytes!("../../assets/toy_text/C8.png"),
            (b'C', b'9') => include_bytes!("../../assets/toy_text/C9.png"),
            (b'C', b'T') => include_bytes!("../../assets/toy_text/CT.png"),
            (b'C', b'J') => include_bytes!("../../assets/toy_text/CJ.png"),
            (b'C', b'Q') => include_bytes!("../../assets/toy_text/CQ.png"),
            (b'C', b'K') => include_bytes!("../../assets/toy_text/CK.png"),
            (b'C', b'A') => include_bytes!("../../assets/toy_text/CA.png"),
            (b'D', b'2') => include_bytes!("../../assets/toy_text/D2.png"),
            (b'D', b'3') => include_bytes!("../../assets/toy_text/D3.png"),
            (b'D', b'4') => include_bytes!("../../assets/toy_text/D4.png"),
            (b'D', b'5') => include_bytes!("../../assets/toy_text/D5.png"),
            (b'D', b'6') => include_bytes!("../../assets/toy_text/D6.png"),
            (b'D', b'7') => include_bytes!("../../assets/toy_text/D7.png"),
            (b'D', b'8') => include_bytes!("../../assets/toy_text/D8.png"),
            (b'D', b'9') => include_bytes!("../../assets/toy_text/D9.png"),
            (b'D', b'T') => include_bytes!("../../assets/toy_text/DT.png"),
            (b'D', b'J') => include_bytes!("../../assets/toy_text/DJ.png"),
            (b'D', b'Q') => include_bytes!("../../assets/toy_text/DQ.png"),
            (b'D', b'K') => include_bytes!("../../assets/toy_text/DK.png"),
            (b'D', b'A') => include_bytes!("../../assets/toy_text/DA.png"),
            (b'H', b'2') => include_bytes!("../../assets/toy_text/H2.png"),
            (b'H', b'3') => include_bytes!("../../assets/toy_text/H3.png"),
            (b'H', b'4') => include_bytes!("../../assets/toy_text/H4.png"),
            (b'H', b'5') => include_bytes!("../../assets/toy_text/H5.png"),
            (b'H', b'6') => include_bytes!("../../assets/toy_text/H6.png"),
            (b'H', b'7') => include_bytes!("../../assets/toy_text/H7.png"),
            (b'H', b'8') => include_bytes!("../../assets/toy_text/H8.png"),
            (b'H', b'9') => include_bytes!("../../assets/toy_text/H9.png"),
            (b'H', b'T') => include_bytes!("../../assets/toy_text/HT.png"),
            (b'H', b'J') => include_bytes!("../../assets/toy_text/HJ.png"),
            (b'H', b'Q') => include_bytes!("../../assets/toy_text/HQ.png"),
            (b'H', b'K') => include_bytes!("../../assets/toy_text/HK.png"),
            (b'H', b'A') => include_bytes!("../../assets/toy_text/HA.png"),
            (b'S', b'2') => include_bytes!("../../assets/toy_text/S2.png"),
            (b'S', b'3') => include_bytes!("../../assets/toy_text/S3.png"),
            (b'S', b'4') => include_bytes!("../../assets/toy_text/S4.png"),
            (b'S', b'5') => include_bytes!("../../assets/toy_text/S5.png"),
            (b'S', b'6') => include_bytes!("../../assets/toy_text/S6.png"),
            (b'S', b'7') => include_bytes!("../../assets/toy_text/S7.png"),
            (b'S', b'8') => include_bytes!("../../assets/toy_text/S8.png"),
            (b'S', b'9') => include_bytes!("../../assets/toy_text/S9.png"),
            (b'S', b'T') => include_bytes!("../../assets/toy_text/ST.png"),
            (b'S', b'J') => include_bytes!("../../assets/toy_text/SJ.png"),
            (b'S', b'Q') => include_bytes!("../../assets/toy_text/SQ.png"),
            (b'S', b'K') => include_bytes!("../../assets/toy_text/SK.png"),
            (b'S', b'A') => include_bytes!("../../assets/toy_text/SA.png"),
            _ => include_bytes!("../../assets/toy_text/Card.png"),
        }
    }
}

/// Pre-decoded and scaled sprites for `FrozenLake`.
pub struct FrozenLakeSprites {
    /// Ice tile.
    pub ice: Pixmap,
    /// Hole tile.
    pub hole: Pixmap,
    /// Cracked hole (elf fell in).
    pub cracked_hole: Pixmap,
    /// Goal tile.
    pub goal: Pixmap,
    /// Start stool.
    pub stool: Pixmap,
    /// Elf images: \[left, down, right, up\].
    pub elf: [Pixmap; 4],
}

impl std::fmt::Debug for FrozenLakeSprites {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrozenLakeSprites").finish_non_exhaustive()
    }
}

impl FrozenLakeSprites {
    /// Decode and scale all `FrozenLake` sprites to `(cw, ch)` cell size.
    #[must_use]
    pub fn new(cw: u32, ch: u32) -> Self {
        Self {
            ice: Canvas::decode_png_scaled(frozen_lake::ICE, cw, ch),
            hole: Canvas::decode_png_scaled(frozen_lake::HOLE, cw, ch),
            cracked_hole: Canvas::decode_png_scaled(frozen_lake::CRACKED_HOLE, cw, ch),
            goal: Canvas::decode_png_scaled(frozen_lake::GOAL, cw, ch),
            stool: Canvas::decode_png_scaled(frozen_lake::STOOL, cw, ch),
            elf: [
                Canvas::decode_png_scaled(frozen_lake::ELF_LEFT, cw, ch),
                Canvas::decode_png_scaled(frozen_lake::ELF_DOWN, cw, ch),
                Canvas::decode_png_scaled(frozen_lake::ELF_RIGHT, cw, ch),
                Canvas::decode_png_scaled(frozen_lake::ELF_UP, cw, ch),
            ],
        }
    }
}

/// Pre-decoded and scaled sprites for `CliffWalking`.
pub struct CliffWalkingSprites {
    /// Mountain background variants (checkerboard).
    pub bg: [Pixmap; 2],
    /// Cliff tile.
    pub cliff: Pixmap,
    /// Near-cliff variants.
    pub near_cliff: [Pixmap; 2],
    /// Start stool.
    pub stool: Pixmap,
    /// Goal cookie.
    pub cookie: Pixmap,
    /// Elf images: \[up, right, down, left\].
    pub elf: [Pixmap; 4],
}

impl std::fmt::Debug for CliffWalkingSprites {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CliffWalkingSprites")
            .finish_non_exhaustive()
    }
}

impl CliffWalkingSprites {
    /// Decode and scale all `CliffWalking` sprites to `(cw, ch)` cell size.
    #[must_use]
    pub fn new(cw: u32, ch: u32) -> Self {
        Self {
            bg: [
                Canvas::decode_png_scaled(cliff_walking::BG1, cw, ch),
                Canvas::decode_png_scaled(cliff_walking::BG2, cw, ch),
            ],
            cliff: Canvas::decode_png_scaled(cliff_walking::CLIFF, cw, ch),
            near_cliff: [
                Canvas::decode_png_scaled(cliff_walking::NEAR_CLIFF1, cw, ch),
                Canvas::decode_png_scaled(cliff_walking::NEAR_CLIFF2, cw, ch),
            ],
            stool: Canvas::decode_png_scaled(frozen_lake::STOOL, cw, ch),
            cookie: Canvas::decode_png_scaled(cliff_walking::COOKIE, cw, ch),
            elf: [
                Canvas::decode_png_scaled(frozen_lake::ELF_UP, cw, ch),
                Canvas::decode_png_scaled(frozen_lake::ELF_RIGHT, cw, ch),
                Canvas::decode_png_scaled(frozen_lake::ELF_DOWN, cw, ch),
                Canvas::decode_png_scaled(frozen_lake::ELF_LEFT, cw, ch),
            ],
        }
    }
}

/// Pre-decoded and scaled sprites for Taxi.
pub struct TaxiSprites {
    /// Background tile.
    pub bg: Pixmap,
    /// Cab images: \[front(south), rear(north), right(east), left(west)\].
    pub cab: [Pixmap; 4],
    /// Passenger sprite.
    pub passenger: Pixmap,
    /// Destination hotel sprite.
    pub hotel: Pixmap,
    /// Horizontal median: \[left, middle, right\].
    pub median_horiz: [Pixmap; 3],
    /// Vertical median: \[top, middle, bottom\].
    pub median_vert: [Pixmap; 3],
}

impl std::fmt::Debug for TaxiSprites {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaxiSprites").finish_non_exhaustive()
    }
}

impl TaxiSprites {
    /// Decode and scale all Taxi sprites to `(cw, ch)` cell size.
    #[must_use]
    pub fn new(cw: u32, ch: u32) -> Self {
        Self {
            bg: Canvas::decode_png_scaled(taxi::BACKGROUND, cw, ch),
            cab: [
                Canvas::decode_png_scaled(taxi::CAB_FRONT, cw, ch),
                Canvas::decode_png_scaled(taxi::CAB_REAR, cw, ch),
                Canvas::decode_png_scaled(taxi::CAB_RIGHT, cw, ch),
                Canvas::decode_png_scaled(taxi::CAB_LEFT, cw, ch),
            ],
            passenger: Canvas::decode_png_scaled(taxi::PASSENGER, cw, ch),
            hotel: Canvas::decode_png_scaled(taxi::HOTEL, cw, ch),
            median_horiz: [
                Canvas::decode_png_scaled(taxi::MEDIAN_LEFT, cw, ch),
                Canvas::decode_png_scaled(taxi::MEDIAN_HORIZ, cw, ch),
                Canvas::decode_png_scaled(taxi::MEDIAN_RIGHT, cw, ch),
            ],
            median_vert: [
                Canvas::decode_png_scaled(taxi::MEDIAN_TOP, cw, ch),
                Canvas::decode_png_scaled(taxi::MEDIAN_VERT, cw, ch),
                Canvas::decode_png_scaled(taxi::MEDIAN_BOTTOM, cw, ch),
            ],
        }
    }
}

/// Pre-decoded and scaled card back for Blackjack.
pub struct BlackjackSprites {
    /// Card back image.
    pub card_back: Pixmap,
}

impl std::fmt::Debug for BlackjackSprites {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlackjackSprites").finish_non_exhaustive()
    }
}

impl BlackjackSprites {
    /// Decode and scale the card-back sprite to `(cw, ch)`.
    #[must_use]
    pub fn new(cw: u32, ch: u32) -> Self {
        Self {
            card_back: Canvas::decode_png_scaled(blackjack::CARD_BACK, cw, ch),
        }
    }

    /// Decode and scale a specific card face.
    #[must_use]
    pub fn decode_card(suit: u8, value: u8, cw: u32, ch: u32) -> Pixmap {
        Canvas::decode_png_scaled(blackjack::card_png(suit, value), cw, ch)
    }
}
