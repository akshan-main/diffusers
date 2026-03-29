# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure utility functions for tiled upscale workflows.

Supports:
- Linear (raster) and chess (checkerboard) tile traversal
- Non-overlapping core paste and gradient overlap blending
- Seam-fix band planning along tile boundaries
- Linear feathered mask blending for seam-fix bands
"""

from dataclasses import dataclass, field

import numpy as np
import PIL.Image


@dataclass
class TileSpec:
    """Specification for a single tile, distinguishing the core output region
    from the padded crop region used for denoising.

    Attributes:
        core_x: Left edge of the core region in the output canvas.
        core_y: Top edge of the core region in the output canvas.
        core_w: Width of the core region (what this tile is responsible for pasting).
        core_h: Height of the core region.
        crop_x: Left edge of the padded crop region in the source image.
        crop_y: Top edge of the padded crop region in the source image.
        crop_w: Width of the padded crop region (what gets denoised).
        crop_h: Height of the padded crop region.
        paste_x: X offset of the core region within the crop (left padding amount).
        paste_y: Y offset of the core region within the crop (top padding amount).
    """

    core_x: int
    core_y: int
    core_w: int
    core_h: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    paste_x: int
    paste_y: int


@dataclass
class SeamFixSpec:
    """Specification for a seam-fix band along a tile boundary.

    Attributes:
        band_x: Left edge of the band in the output canvas.
        band_y: Top edge of the band in the output canvas.
        band_w: Width of the band.
        band_h: Height of the band.
        crop_x: Left edge of the padded crop for denoising.
        crop_y: Top edge of the padded crop for denoising.
        crop_w: Width of the padded crop.
        crop_h: Height of the padded crop.
        paste_x: X offset of the band within the crop.
        paste_y: Y offset of the band within the crop.
        orientation: 'horizontal' or 'vertical'.
    """

    band_x: int
    band_y: int
    band_w: int
    band_h: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    paste_x: int
    paste_y: int
    orientation: str = field(default="horizontal")


def validate_tile_params(tile_size: int, tile_padding: int) -> None:
    """Validate tile parameters strictly.

    Args:
        tile_size: Base tile size in pixels.
        tile_padding: Overlap padding on each side.

    Raises:
        ValueError: If parameters are out of range.
    """
    if tile_size <= 0:
        raise ValueError(f"`tile_size` must be positive, got {tile_size}.")
    if tile_padding < 0:
        raise ValueError(f"`tile_padding` must be non-negative, got {tile_padding}.")
    if tile_padding >= tile_size // 2:
        raise ValueError(
            f"`tile_padding` must be less than tile_size // 2. "
            f"Got tile_padding={tile_padding}, tile_size={tile_size} "
            f"(max allowed: {tile_size // 2 - 1})."
        )


def plan_tiles_linear(
    image_width: int,
    image_height: int,
    tile_size: int = 512,
    tile_padding: int = 32,
) -> list[TileSpec]:
    """Plan tiles in a left-to-right, top-to-bottom (linear/raster) traversal order.

    Each tile is a ``TileSpec`` with separate core (output responsibility) and
    crop (denoised region with padding context) bounds. The crop region extends
    beyond the core by ``tile_padding`` on each side, clamped to image edges.

    Args:
        image_width: Width of the image to tile.
        image_height: Height of the image to tile.
        tile_size: Base tile size. The core region of each tile is
            ``tile_size - 2 * tile_padding``.
        tile_padding: Number of overlap pixels on each side.

    Returns:
        List of ``TileSpec`` in linear traversal order.
    """
    validate_tile_params(tile_size, tile_padding)

    core_size = tile_size - 2 * tile_padding
    tiles: list[TileSpec] = []

    core_y = 0
    while core_y < image_height:
        core_h = min(core_size, image_height - core_y)

        core_x = 0
        while core_x < image_width:
            core_w = min(core_size, image_width - core_x)

            # Compute padded crop region, clamped to image bounds
            crop_x = max(0, core_x - tile_padding)
            crop_y = max(0, core_y - tile_padding)
            crop_x2 = min(image_width, core_x + core_w + tile_padding)
            crop_y2 = min(image_height, core_y + core_h + tile_padding)
            crop_w = crop_x2 - crop_x
            crop_h = crop_y2 - crop_y

            # Where the core sits within the crop
            paste_x = core_x - crop_x
            paste_y = core_y - crop_y

            tiles.append(
                TileSpec(
                    core_x=core_x,
                    core_y=core_y,
                    core_w=core_w,
                    core_h=core_h,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    paste_x=paste_x,
                    paste_y=paste_y,
                )
            )

            core_x += core_size
        core_y += core_size

    return tiles


def crop_tile(image: PIL.Image.Image, tile: TileSpec) -> PIL.Image.Image:
    """Crop the padded region of a tile from a PIL image.

    Args:
        image: Source image.
        tile: Tile specification.

    Returns:
        Cropped PIL image of the padded crop region.
    """
    return image.crop((tile.crop_x, tile.crop_y, tile.crop_x + tile.crop_w, tile.crop_y + tile.crop_h))


def extract_core_from_decoded(decoded_image: np.ndarray, tile: TileSpec) -> np.ndarray:
    """Extract the core region from a decoded tile image.

    Args:
        decoded_image: Decoded tile as numpy array, shape (crop_h, crop_w, C).
        tile: Tile specification.

    Returns:
        Core region as numpy array, shape (core_h, core_w, C).
    """
    return decoded_image[
        tile.paste_y : tile.paste_y + tile.core_h,
        tile.paste_x : tile.paste_x + tile.core_w,
    ]


def paste_core_into_canvas(
    canvas: np.ndarray,
    core_image: np.ndarray,
    tile: TileSpec,
) -> None:
    """Paste the core region of a decoded tile directly into the output canvas.

    No blending — the core regions tile the canvas without overlap.

    Args:
        canvas: Output canvas, shape (H, W, C), float32. Modified in-place.
        core_image: Core tile pixels, shape (core_h, core_w, C), float32.
        tile: Tile specification.
    """
    canvas[tile.core_y : tile.core_y + tile.core_h, tile.core_x : tile.core_x + tile.core_w] = core_image


# =============================================================================
# Chess (checkerboard) traversal
# =============================================================================


def plan_tiles_chess(
    image_width: int,
    image_height: int,
    tile_size: int = 512,
    tile_padding: int = 32,
) -> list[TileSpec]:
    """Plan tiles in a checkerboard (chess) traversal order.

    Two passes: first all "white" tiles (row+col both even or both odd),
    then all "black" tiles. This ensures adjacent tiles are never processed
    consecutively, reducing visible seam patterns.

    Args:
        image_width: Width of the image to tile.
        image_height: Height of the image to tile.
        tile_size: Base tile size.
        tile_padding: Number of overlap pixels on each side.

    Returns:
        List of ``TileSpec`` in chess traversal order.
    """
    validate_tile_params(tile_size, tile_padding)

    core_size = tile_size - 2 * tile_padding

    # Build grid of all tiles with (row, col) indices
    grid: list[tuple[int, int, TileSpec]] = []

    row = 0
    core_y = 0
    while core_y < image_height:
        core_h = min(core_size, image_height - core_y)

        col = 0
        core_x = 0
        while core_x < image_width:
            core_w = min(core_size, image_width - core_x)

            crop_x = max(0, core_x - tile_padding)
            crop_y = max(0, core_y - tile_padding)
            crop_x2 = min(image_width, core_x + core_w + tile_padding)
            crop_y2 = min(image_height, core_y + core_h + tile_padding)
            crop_w = crop_x2 - crop_x
            crop_h = crop_y2 - crop_y

            paste_x = core_x - crop_x
            paste_y = core_y - crop_y

            tile = TileSpec(
                core_x=core_x, core_y=core_y, core_w=core_w, core_h=core_h,
                crop_x=crop_x, crop_y=crop_y, crop_w=crop_w, crop_h=crop_h,
                paste_x=paste_x, paste_y=paste_y,
            )
            grid.append((row, col, tile))

            col += 1
            core_x += core_size
        row += 1
        core_y += core_size

    # Separate into white and black squares
    white = [t for r, c, t in grid if (r + c) % 2 == 0]
    black = [t for r, c, t in grid if (r + c) % 2 == 1]

    return white + black


# =============================================================================
# Gradient overlap blending
# =============================================================================


def make_gradient_mask(
    core_h: int,
    core_w: int,
    overlap: int,
    at_top: bool = False,
    at_bottom: bool = False,
    at_left: bool = False,
    at_right: bool = False,
) -> np.ndarray:
    """Create a boundary-aware gradient blending mask for a tile's core region.

    The mask is 1.0 in the interior and linearly ramps from 0 to 1 in the
    overlap zones along interior edges only. Edges that touch the canvas
    boundary (indicated by ``at_*`` flags) stay at 1.0 to prevent black borders.

    Args:
        core_h: Height of the core region.
        core_w: Width of the core region.
        overlap: Width of the gradient ramp in pixels.
        at_top: True if tile is at the top edge of the canvas.
        at_bottom: True if tile is at the bottom edge of the canvas.
        at_left: True if tile is at the left edge of the canvas.
        at_right: True if tile is at the right edge of the canvas.

    Returns:
        Mask of shape (core_h, core_w), float32, values in [0, 1].
    """
    if overlap <= 0:
        return np.ones((core_h, core_w), dtype=np.float32)

    mask = np.ones((core_h, core_w), dtype=np.float32)

    # Only fade on interior edges (not canvas boundaries)
    ramp_w = min(overlap, core_w)
    if ramp_w > 0 and not at_left:
        left_ramp = np.linspace(0.0, 1.0, ramp_w, dtype=np.float32)
        mask[:, :ramp_w] = np.minimum(mask[:, :ramp_w], left_ramp[np.newaxis, :])
    if ramp_w > 0 and not at_right:
        right_ramp = np.linspace(1.0, 0.0, ramp_w, dtype=np.float32)
        mask[:, -ramp_w:] = np.minimum(mask[:, -ramp_w:], right_ramp[np.newaxis, :])

    ramp_h = min(overlap, core_h)
    if ramp_h > 0 and not at_top:
        top_ramp = np.linspace(0.0, 1.0, ramp_h, dtype=np.float32)
        mask[:ramp_h, :] = np.minimum(mask[:ramp_h, :], top_ramp[:, np.newaxis])
    if ramp_h > 0 and not at_bottom:
        bottom_ramp = np.linspace(1.0, 0.0, ramp_h, dtype=np.float32)
        mask[-ramp_h:, :] = np.minimum(mask[-ramp_h:, :], bottom_ramp[:, np.newaxis])

    return mask


def paste_core_into_canvas_blended(
    canvas: np.ndarray,
    weight_map: np.ndarray,
    core_image: np.ndarray,
    tile: TileSpec,
    overlap: int,
) -> None:
    """Paste a tile's core into the canvas using boundary-aware gradient blending.

    Uses accumulated weighted sum approach: canvas stores weighted sum,
    weight_map stores total weights. Finalize by dividing canvas / weight_map.

    Args:
        canvas: Output canvas, shape (H, W, C), float32. Modified in-place.
        weight_map: Weight accumulator, shape (H, W), float32. Modified in-place.
        core_image: Core tile pixels, shape (core_h, core_w, C), float32.
        tile: Tile specification.
        overlap: Gradient overlap width in pixels.
    """
    canvas_h, canvas_w = canvas.shape[:2]

    mask = make_gradient_mask(
        tile.core_h, tile.core_w, overlap,
        at_top=(tile.core_y == 0),
        at_bottom=(tile.core_y + tile.core_h >= canvas_h),
        at_left=(tile.core_x == 0),
        at_right=(tile.core_x + tile.core_w >= canvas_w),
    )

    y1, y2 = tile.core_y, tile.core_y + tile.core_h
    x1, x2 = tile.core_x, tile.core_x + tile.core_w

    canvas[y1:y2, x1:x2] += core_image * mask[:, :, np.newaxis]
    weight_map[y1:y2, x1:x2] += mask


def finalize_blended_canvas(canvas: np.ndarray, weight_map: np.ndarray) -> np.ndarray:
    """Normalize the blended canvas by dividing by accumulated weights.

    Pixels with zero weight (uncovered) are filled from the raw weighted sum
    to avoid black borders from epsilon division.

    Args:
        canvas: Weighted sum canvas, shape (H, W, C).
        weight_map: Weight accumulator, shape (H, W).

    Returns:
        Normalized canvas, shape (H, W, C), float32.
    """
    result = np.copy(canvas)
    covered = weight_map > 0
    result[covered] = canvas[covered] / weight_map[covered, np.newaxis]
    # Uncovered pixels stay as-is (zero) — should not occur with boundary-aware masks
    return result


# =============================================================================
# Seam-fix band planning
# =============================================================================


def plan_seam_fix_bands(
    tiles: list[TileSpec],
    image_width: int,
    image_height: int,
    seam_fix_width: int = 64,
    seam_fix_padding: int = 16,
) -> list[SeamFixSpec]:
    """Plan seam-fix bands along tile boundaries.

    For each pair of adjacent core regions, creates a band centered on the
    shared boundary. Bands are denoised in a second pass to smooth seams.

    Args:
        tiles: The tile plan (from plan_tiles_linear or plan_tiles_chess).
        image_width: Full image width.
        image_height: Full image height.
        seam_fix_width: Width of the seam-fix band in pixels.
        seam_fix_padding: Additional padding around each band for denoise context.

    Returns:
        List of ``SeamFixSpec`` for all seam boundaries.
    """
    if seam_fix_width < 0:
        raise ValueError(f"`seam_fix_width` must be non-negative, got {seam_fix_width}.")
    if seam_fix_width == 0:
        return []
    if seam_fix_padding < 0:
        raise ValueError(f"`seam_fix_padding` must be non-negative, got {seam_fix_padding}.")

    # Collect unique boundary positions
    h_boundaries: set[tuple[int, int, int]] = set()  # (y, x_start, x_end)
    v_boundaries: set[tuple[int, int, int]] = set()  # (x, y_start, y_end)

    for tile in tiles:
        # Bottom edge of this tile → horizontal seam
        bottom_y = tile.core_y + tile.core_h
        if bottom_y < image_height:
            h_boundaries.add((bottom_y, tile.core_x, tile.core_x + tile.core_w))

        # Right edge → vertical seam
        right_x = tile.core_x + tile.core_w
        if right_x < image_width:
            v_boundaries.add((right_x, tile.core_y, tile.core_y + tile.core_h))

    bands: list[SeamFixSpec] = []
    half_left = seam_fix_width // 2
    half_right = seam_fix_width - half_left

    for y, x_start, x_end in sorted(h_boundaries):
        band_y = max(0, y - half_left)
        band_y2 = min(image_height, y + half_right)
        band_h = band_y2 - band_y
        band_w = x_end - x_start

        crop_x = max(0, x_start - seam_fix_padding)
        crop_y = max(0, band_y - seam_fix_padding)
        crop_x2 = min(image_width, x_end + seam_fix_padding)
        crop_y2 = min(image_height, band_y2 + seam_fix_padding)

        bands.append(SeamFixSpec(
            band_x=x_start, band_y=band_y, band_w=band_w, band_h=band_h,
            crop_x=crop_x, crop_y=crop_y,
            crop_w=crop_x2 - crop_x, crop_h=crop_y2 - crop_y,
            paste_x=x_start - crop_x, paste_y=band_y - crop_y,
            orientation="horizontal",
        ))

    for x, y_start, y_end in sorted(v_boundaries):
        band_x = max(0, x - half_left)
        band_x2 = min(image_width, x + half_right)
        band_w = band_x2 - band_x
        band_h = y_end - y_start

        crop_x = max(0, band_x - seam_fix_padding)
        crop_y = max(0, y_start - seam_fix_padding)
        crop_x2 = min(image_width, band_x2 + seam_fix_padding)
        crop_y2 = min(image_height, y_end + seam_fix_padding)

        bands.append(SeamFixSpec(
            band_x=band_x, band_y=y_start, band_w=band_w, band_h=band_h,
            crop_x=crop_x, crop_y=crop_y,
            crop_w=crop_x2 - crop_x, crop_h=crop_y2 - crop_y,
            paste_x=band_x - crop_x, paste_y=y_start - crop_y,
            orientation="vertical",
        ))

    return bands


def extract_band_from_decoded(decoded_image: np.ndarray, band: SeamFixSpec) -> np.ndarray:
    """Extract the band region from a decoded seam-fix image."""
    return decoded_image[
        band.paste_y : band.paste_y + band.band_h,
        band.paste_x : band.paste_x + band.band_w,
    ]


def make_seam_fix_mask(band: SeamFixSpec, mask_blur: int = 8) -> np.ndarray:
    """Create a linearly-feathered mask for a seam-fix band.

    The mask is 1.0 at the center of the seam and linearly fades to 0.0
    at the edges perpendicular to the seam orientation, so the seam-fix
    blends smoothly with the surrounding tile results.

    Args:
        band: Seam-fix band specification.
        mask_blur: Width of the linear feather ramp in pixels.

    Returns:
        Mask of shape (band_h, band_w), float32, values in [0, 1].
    """
    if mask_blur <= 0:
        return np.ones((band.band_h, band.band_w), dtype=np.float32)

    mask = np.ones((band.band_h, band.band_w), dtype=np.float32)

    if band.orientation == "horizontal":
        # Fade along height (top/bottom edges)
        ramp = min(mask_blur, band.band_h // 2)
        if ramp > 0:
            top_ramp = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
            mask[:ramp, :] = top_ramp[:, np.newaxis]
            bottom_ramp = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
            mask[-ramp:, :] = bottom_ramp[:, np.newaxis]
    else:
        # Fade along width (left/right edges)
        ramp = min(mask_blur, band.band_w // 2)
        if ramp > 0:
            left_ramp = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
            mask[:, :ramp] = left_ramp[np.newaxis, :]
            right_ramp = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
            mask[:, -ramp:] = right_ramp[np.newaxis, :]

    return mask


def paste_seam_fix_band(
    canvas: np.ndarray,
    band_image: np.ndarray,
    band: SeamFixSpec,
    mask_blur: int = 8,
) -> None:
    """Paste a seam-fix band into the canvas with feathered blending.

    Args:
        canvas: Output canvas, shape (H, W, C), float32. Modified in-place.
        band_image: Decoded band pixels, shape (band_h, band_w, C), float32.
        band: Seam-fix band specification.
        mask_blur: Feathering width.
    """
    mask = make_seam_fix_mask(band, mask_blur)

    y1, y2 = band.band_y, band.band_y + band.band_h
    x1, x2 = band.band_x, band.band_x + band.band_w

    existing = canvas[y1:y2, x1:x2]
    canvas[y1:y2, x1:x2] = existing * (1 - mask[:, :, np.newaxis]) + band_image * mask[:, :, np.newaxis]


# =============================================================================
# Latent-space tile planning for MultiDiffusion
# =============================================================================


@dataclass
class LatentTileSpec:
    """Tile specification in latent space for MultiDiffusion.

    Attributes:
        y: Top edge in latent pixels.
        x: Left edge in latent pixels.
        h: Height in latent pixels.
        w: Width in latent pixels.
    """

    y: int
    x: int
    h: int
    w: int


def plan_latent_tiles(
    latent_h: int,
    latent_w: int,
    tile_size: int = 64,
    overlap: int = 8,
) -> list[LatentTileSpec]:
    """Plan overlapping tiles in latent space for MultiDiffusion.

    Tiles overlap by ``overlap`` latent pixels. The stride is
    ``tile_size - overlap``. Edge tiles are clamped to the latent bounds.

    Args:
        latent_h: Height of the full latent tensor.
        latent_w: Width of the full latent tensor.
        tile_size: Tile size in latent pixels (e.g., 64 = 512px at scale 8).
        overlap: Overlap in latent pixels (e.g., 8 = 64px at scale 8).

    Returns:
        List of ``LatentTileSpec``.
    """
    if tile_size <= 0:
        raise ValueError(f"`tile_size` must be positive, got {tile_size}.")
    if overlap < 0:
        raise ValueError(f"`overlap` must be non-negative, got {overlap}.")
    if overlap >= tile_size:
        raise ValueError(
            f"`overlap` must be less than `tile_size`. "
            f"Got overlap={overlap}, tile_size={tile_size}."
        )

    stride = tile_size - overlap
    tiles: list[LatentTileSpec] = []

    y = 0
    while y < latent_h:
        h = min(tile_size, latent_h - y)
        # If remaining height is less than tile_size, shift back to get a full tile
        if h < tile_size and y > 0:
            y = max(0, latent_h - tile_size)
            h = latent_h - y

        x = 0
        while x < latent_w:
            w = min(tile_size, latent_w - x)
            if w < tile_size and x > 0:
                x = max(0, latent_w - tile_size)
                w = latent_w - x

            tiles.append(LatentTileSpec(y=y, x=x, h=h, w=w))

            if x + w >= latent_w:
                break
            x += stride

        if y + h >= latent_h:
            break
        y += stride

    return tiles
