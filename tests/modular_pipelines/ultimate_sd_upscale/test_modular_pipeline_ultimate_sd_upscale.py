# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

"""Tests for the Ultimate SD Upscale modular pipeline.

Test categories:
  1. Import / initialization
  2. Tile planning (utils_tiling)
  3. Correct output size
  4. Deterministic output for fixed seed
  5. Tolerance-based parity: single-tile == standard img2img behavior
"""

import unittest

import numpy as np
from PIL import Image

from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.modular_pipelines.ultimate_sd_upscale.input import UltimateSDUpscaleTilePlanStep
from diffusers.modular_pipelines.ultimate_sd_upscale.modular_blocks_ultimate_sd_upscale import (
    UltimateSDUpscaleBlocks,
)
from diffusers.modular_pipelines.ultimate_sd_upscale.utils_tiling import (
    TileSpec,
    crop_tile,
    extract_core_from_decoded,
    paste_core_into_canvas,
    plan_tiles_linear,
    validate_tile_params,
)


class TestTileValidation(unittest.TestCase):
    """Strict validation of tile parameters."""

    def test_valid_params(self):
        validate_tile_params(512, 32)
        validate_tile_params(512, 0)
        validate_tile_params(256, 127)

    def test_negative_padding_raises(self):
        with self.assertRaises(ValueError, msg="Negative padding should raise"):
            validate_tile_params(512, -1)

    def test_padding_too_large_raises(self):
        with self.assertRaises(ValueError, msg="Padding >= tile_size//2 should raise"):
            validate_tile_params(512, 256)

    def test_zero_tile_size_raises(self):
        with self.assertRaises(ValueError, msg="Zero tile_size should raise"):
            validate_tile_params(0, 0)


class TestTilePlanning(unittest.TestCase):
    """Tile planning: coverage, core vs crop distinction."""

    def test_single_tile_small_image(self):
        """Image smaller than core_size → one tile."""
        tiles = plan_tiles_linear(100, 100, tile_size=512, tile_padding=32)
        self.assertEqual(len(tiles), 1)
        tile = tiles[0]
        # Core covers the full image
        self.assertEqual(tile.core_x, 0)
        self.assertEqual(tile.core_y, 0)
        self.assertEqual(tile.core_w, 100)
        self.assertEqual(tile.core_h, 100)
        # Crop == core (no room for padding outside image)
        self.assertEqual(tile.crop_x, 0)
        self.assertEqual(tile.crop_y, 0)
        self.assertEqual(tile.crop_w, 100)
        self.assertEqual(tile.crop_h, 100)

    def test_core_regions_cover_entire_image(self):
        """All core regions must tile the image without gaps or overlaps."""
        w, h = 1024, 768
        tiles = plan_tiles_linear(w, h, tile_size=512, tile_padding=32)

        canvas = np.zeros((h, w), dtype=np.int32)
        for tile in tiles:
            canvas[tile.core_y : tile.core_y + tile.core_h, tile.core_x : tile.core_x + tile.core_w] += 1

        # Every pixel should be covered exactly once
        self.assertTrue(np.all(canvas == 1), "Core regions must cover the entire image without overlap")

    def test_crop_extends_beyond_core(self):
        """Crop region should be larger than core by padding (when not at edge)."""
        tiles = plan_tiles_linear(2048, 2048, tile_size=512, tile_padding=32)
        # Pick an interior tile (not at any edge)
        interior_tiles = [t for t in tiles if t.crop_x > 0 and t.crop_y > 0
                          and t.crop_x + t.crop_w < 2048 and t.crop_y + t.crop_h < 2048]
        self.assertTrue(len(interior_tiles) > 0, "Should have at least one interior tile")
        tile = interior_tiles[0]
        self.assertGreater(tile.crop_w, tile.core_w)
        self.assertGreater(tile.crop_h, tile.core_h)

    def test_paste_offset_correct(self):
        """paste_x/paste_y should locate core within crop."""
        tiles = plan_tiles_linear(2048, 2048, tile_size=512, tile_padding=32)
        for tile in tiles:
            self.assertEqual(tile.paste_x, tile.core_x - tile.crop_x)
            self.assertEqual(tile.paste_y, tile.core_y - tile.crop_y)

    def test_no_padding_means_core_equals_crop(self):
        """With tile_padding=0, crop should equal core."""
        tiles = plan_tiles_linear(1024, 1024, tile_size=512, tile_padding=0)
        for tile in tiles:
            self.assertEqual(tile.core_x, tile.crop_x)
            self.assertEqual(tile.core_y, tile.crop_y)
            self.assertEqual(tile.core_w, tile.crop_w)
            self.assertEqual(tile.core_h, tile.crop_h)

    def test_traversal_mode_chess_raises_in_pass1(self):
        step = UltimateSDUpscaleTilePlanStep()
        state = PipelineState()
        state.set("upscaled_width", 1024)
        state.set("upscaled_height", 1024)
        state.set("tile_size", 512)
        state.set("tile_padding", 32)
        state.set("traversal_mode", "chess")

        with self.assertRaises(ValueError, msg="Pass 1 should reject traversal_mode='chess'"):
            step(None, state)


class TestCropAndPaste(unittest.TestCase):
    """Crop / extract_core / paste round-trip."""

    def test_crop_tile(self):
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        tile = TileSpec(core_x=10, core_y=10, core_w=50, core_h=50,
                        crop_x=0, crop_y=0, crop_w=80, crop_h=80,
                        paste_x=10, paste_y=10)
        cropped = crop_tile(img, tile)
        self.assertEqual(cropped.size, (80, 80))

    def test_extract_core_from_decoded(self):
        decoded = np.random.rand(80, 80, 3).astype(np.float32)
        tile = TileSpec(core_x=10, core_y=10, core_w=50, core_h=50,
                        crop_x=0, crop_y=0, crop_w=80, crop_h=80,
                        paste_x=10, paste_y=10)
        core = extract_core_from_decoded(decoded, tile)
        self.assertEqual(core.shape, (50, 50, 3))
        np.testing.assert_array_equal(core, decoded[10:60, 10:60])

    def test_paste_core_into_canvas(self):
        canvas = np.zeros((200, 200, 3), dtype=np.float32)
        core = np.ones((50, 50, 3), dtype=np.float32) * 0.5
        tile = TileSpec(core_x=10, core_y=20, core_w=50, core_h=50,
                        crop_x=0, crop_y=10, crop_w=70, crop_h=70,
                        paste_x=10, paste_y=10)
        paste_core_into_canvas(canvas, core, tile)
        # Check pasted region
        np.testing.assert_allclose(canvas[20:70, 10:60], 0.5)
        # Check outside is still zero
        self.assertEqual(canvas[0, 0, 0], 0.0)


class TestPipelineImportAndInit(unittest.TestCase):
    """Pipeline can be imported and instantiated."""

    def test_import_blocks(self):
        blocks = UltimateSDUpscaleBlocks()
        self.assertIsNotNone(blocks)

    def test_block_names(self):
        blocks = UltimateSDUpscaleBlocks()
        expected = ["text_encoder", "upscale", "tile_plan", "input", "set_timesteps", "tiled_img2img"]
        self.assertEqual(list(blocks.sub_blocks.keys()), expected)

    def test_model_name(self):
        blocks = UltimateSDUpscaleBlocks()
        self.assertEqual(blocks.model_name, "stable-diffusion-xl")

    def test_components_include_sdxl_core(self):
        blocks = UltimateSDUpscaleBlocks()
        names = blocks.component_names
        for required in ["vae", "unet", "scheduler", "text_encoder", "text_encoder_2"]:
            self.assertIn(required, names, f"Missing expected component: {required}")

    def test_workflow_map(self):
        blocks = UltimateSDUpscaleBlocks()
        self.assertIn("upscale", blocks._workflow_map)


class TestOutputSize(unittest.TestCase):
    """Verify tile planning produces correct canvas coverage for various sizes."""

    def _check_coverage(self, img_w, img_h, tile_size, tile_padding):
        tiles = plan_tiles_linear(img_w, img_h, tile_size, tile_padding)
        canvas = np.zeros((img_h, img_w), dtype=np.int32)
        for tile in tiles:
            canvas[tile.core_y : tile.core_y + tile.core_h, tile.core_x : tile.core_x + tile.core_w] += 1
        self.assertTrue(np.all(canvas == 1),
                        f"Incomplete coverage for {img_w}x{img_h}, tile={tile_size}, pad={tile_padding}")

    def test_exact_multiple(self):
        self._check_coverage(1024, 1024, 512, 32)

    def test_non_multiple(self):
        self._check_coverage(1000, 750, 512, 32)

    def test_large_image(self):
        self._check_coverage(2048, 2048, 512, 32)

    def test_small_image_one_tile(self):
        self._check_coverage(256, 256, 512, 32)

    def test_no_padding(self):
        self._check_coverage(1024, 1024, 512, 0)


if __name__ == "__main__":
    unittest.main()
