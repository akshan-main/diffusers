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

"""Pure utility functions for tiled upscale workflows: tile planning, cropping, core paste.

Pass 1 uses non-overlapping core paste: each tile's core region is pasted
directly into the canvas without overlap blending.  The padded crop region
provides denoising context only and is discarded after core extraction.
"""

from dataclasses import dataclass

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
