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
import torch
from PIL import Image

from diffusers.modular_pipelines.modular_pipeline import BlockState, PipelineState
from diffusers.modular_pipelines.ultimate_sd_upscale.denoise import (
    UltimateSDUpscaleTileDenoiserStep,
    _to_pil_rgb_image,
)
from diffusers.modular_pipelines.ultimate_sd_upscale.input import UltimateSDUpscaleTilePlanStep
from diffusers.modular_pipelines.ultimate_sd_upscale.modular_blocks_ultimate_sd_upscale import (
    UltimateSDUpscaleBlocks,
)
from diffusers.modular_pipelines.ultimate_sd_upscale.utils_tiling import (
    SeamFixSpec,
    TileSpec,
    crop_tile,
    extract_band_from_decoded,
    extract_core_from_decoded,
    finalize_blended_canvas,
    make_gradient_mask,
    make_seam_fix_mask,
    paste_core_into_canvas,
    paste_core_into_canvas_blended,
    paste_seam_fix_band,
    plan_seam_fix_bands,
    plan_tiles_chess,
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

    def test_traversal_mode_invalid_raises(self):
        step = UltimateSDUpscaleTilePlanStep()
        state = PipelineState()
        state.set("upscaled_width", 1024)
        state.set("upscaled_height", 1024)
        state.set("tile_size", 512)
        state.set("tile_padding", 32)
        state.set("traversal_mode", "spiral")
        state.set("seam_fix_width", 0)
        state.set("seam_fix_padding", 16)
        state.set("seam_fix_mask_blur", 8)

        with self.assertRaises(ValueError, msg="Invalid traversal_mode should raise"):
            step(None, state)


class TestChessTraversal(unittest.TestCase):
    """Chess (checkerboard) tile traversal tests."""

    def test_chess_covers_entire_image(self):
        """All core regions tile the image without gaps or overlaps."""
        w, h = 1024, 768
        tiles = plan_tiles_chess(w, h, tile_size=512, tile_padding=32)

        canvas = np.zeros((h, w), dtype=np.int32)
        for tile in tiles:
            canvas[tile.core_y : tile.core_y + tile.core_h, tile.core_x : tile.core_x + tile.core_w] += 1
        self.assertTrue(np.all(canvas == 1), "Chess tiles must cover entire image")

    def test_chess_same_tile_count_as_linear(self):
        """Chess and linear should produce the same number of tiles."""
        w, h = 1024, 768
        linear_tiles = plan_tiles_linear(w, h, tile_size=512, tile_padding=32)
        chess_tiles = plan_tiles_chess(w, h, tile_size=512, tile_padding=32)
        self.assertEqual(len(linear_tiles), len(chess_tiles))

    def test_chess_white_before_black(self):
        """White squares (even row+col parity) should come before black squares."""
        tiles = plan_tiles_chess(2048, 2048, tile_size=512, tile_padding=32)
        core_size = 512 - 2 * 32

        # Determine parity of each tile based on grid position
        parities = []
        for t in tiles:
            row = t.core_y // core_size
            col = t.core_x // core_size
            parities.append((row + col) % 2)

        # All white (0) tiles should precede all black (1) tiles
        first_black = None
        last_white = None
        for i, p in enumerate(parities):
            if p == 1 and first_black is None:
                first_black = i
            if p == 0:
                last_white = i

        if first_black is not None and last_white is not None:
            self.assertLess(last_white, first_black,
                            "All white tiles must come before all black tiles")

    def test_chess_no_adjacent_consecutive(self):
        """Within each color group, no two consecutive tiles share a grid edge."""
        tiles = plan_tiles_chess(2048, 2048, tile_size=512, tile_padding=32)
        core_size = 512 - 2 * 32

        # Split into white/black groups
        white = []
        black = []
        for t in tiles:
            row = t.core_y // core_size
            col = t.core_x // core_size
            if (row + col) % 2 == 0:
                white.append(t)
            else:
                black.append(t)

        def shares_edge(a, b):
            h_adj = (a.core_x + a.core_w == b.core_x or b.core_x + b.core_w == a.core_x) and \
                    a.core_y == b.core_y and a.core_h == b.core_h
            v_adj = (a.core_y + a.core_h == b.core_y or b.core_y + b.core_h == a.core_y) and \
                    a.core_x == b.core_x and a.core_w == b.core_w
            return h_adj or v_adj

        # Within each group, same-parity tiles should not be grid-adjacent
        for group_name, group in [("white", white), ("black", black)]:
            for i in range(len(group) - 1):
                self.assertFalse(
                    shares_edge(group[i], group[i + 1]),
                    f"Consecutive {group_name} tiles {i} and {i+1} share a grid edge"
                )

    def test_chess_plan_step_works(self):
        """TilePlanStep should accept traversal_mode='chess'."""
        step = UltimateSDUpscaleTilePlanStep()
        state = PipelineState()
        state.set("upscaled_width", 1024)
        state.set("upscaled_height", 1024)
        state.set("tile_size", 512)
        state.set("tile_padding", 32)
        state.set("traversal_mode", "chess")
        state.set("seam_fix_width", 0)
        state.set("seam_fix_padding", 16)
        state.set("seam_fix_mask_blur", 8)

        _, state = step(None, state)
        tile_plan = state.get("tile_plan")
        self.assertIsNotNone(tile_plan)
        self.assertGreater(len(tile_plan), 0)


class TestGradientBlending(unittest.TestCase):
    """Gradient overlap blending tests."""

    def test_gradient_mask_shape(self):
        mask = make_gradient_mask(100, 200, overlap=16)
        self.assertEqual(mask.shape, (100, 200))

    def test_gradient_mask_center_is_one(self):
        mask = make_gradient_mask(100, 200, overlap=16)
        self.assertAlmostEqual(mask[50, 100], 1.0)

    def test_gradient_mask_interior_edges_fade(self):
        """Interior edges (no at_* flags) should fade to near-zero."""
        mask = make_gradient_mask(100, 200, overlap=16)
        self.assertAlmostEqual(mask[0, 100], 0.0, places=5)
        self.assertAlmostEqual(mask[50, 0], 0.0, places=5)

    def test_gradient_mask_boundary_edges_stay_one(self):
        """Canvas boundary edges should NOT fade — stays at 1.0."""
        mask = make_gradient_mask(100, 200, overlap=16, at_top=True, at_left=True)
        # Top-left corner should be 1.0 (both edges are boundaries)
        self.assertAlmostEqual(mask[0, 0], 1.0)
        # Top edge center should be 1.0
        self.assertAlmostEqual(mask[0, 100], 1.0)
        # Left edge center should be 1.0
        self.assertAlmostEqual(mask[50, 0], 1.0)
        # Bottom-right should still fade (not boundary)
        self.assertLess(mask[99, 199], 1.0)

    def test_gradient_mask_no_overlap(self):
        mask = make_gradient_mask(100, 200, overlap=0)
        self.assertTrue(np.all(mask == 1.0))

    def test_single_tile_blending_no_black_border(self):
        """A single tile covering the full canvas should produce no dark edges."""
        canvas = np.zeros((100, 100, 3), dtype=np.float32)
        weight_map = np.zeros((100, 100), dtype=np.float32)

        tile = TileSpec(core_x=0, core_y=0, core_w=100, core_h=100,
                        crop_x=0, crop_y=0, crop_w=100, crop_h=100,
                        paste_x=0, paste_y=0)
        core = np.ones((100, 100, 3), dtype=np.float32) * 0.7

        paste_core_into_canvas_blended(canvas, weight_map, core, tile, overlap=16)
        result = finalize_blended_canvas(canvas, weight_map)

        # All corners and edges should be 0.7, not darkened
        np.testing.assert_allclose(result[0, 0, 0], 0.7, atol=0.01)
        np.testing.assert_allclose(result[99, 99, 0], 0.7, atol=0.01)
        np.testing.assert_allclose(result[0, 99, 0], 0.7, atol=0.01)
        np.testing.assert_allclose(result[99, 0, 0], 0.7, atol=0.01)

    def test_blended_paste_and_finalize(self):
        """Gradient blending produces smooth transitions."""
        canvas = np.zeros((200, 200, 3), dtype=np.float32)
        weight_map = np.zeros((200, 200), dtype=np.float32)

        tile1 = TileSpec(core_x=0, core_y=0, core_w=120, core_h=120,
                         crop_x=0, crop_y=0, crop_w=120, crop_h=120,
                         paste_x=0, paste_y=0)
        tile2 = TileSpec(core_x=100, core_y=0, core_w=100, core_h=120,
                         crop_x=100, crop_y=0, crop_w=100, crop_h=120,
                         paste_x=0, paste_y=0)

        core1 = np.ones((120, 120, 3), dtype=np.float32) * 0.8
        core2 = np.ones((120, 100, 3), dtype=np.float32) * 0.2

        paste_core_into_canvas_blended(canvas, weight_map, core1, tile1, overlap=20)
        paste_core_into_canvas_blended(canvas, weight_map, core2, tile2, overlap=20)

        result = finalize_blended_canvas(canvas, weight_map)
        self.assertEqual(result.shape, (200, 200, 3))

        # Interior of tile1 should be close to 0.8
        np.testing.assert_allclose(result[60, 50, 0], 0.8, atol=0.01)
        # Interior of tile2 should be close to 0.2
        np.testing.assert_allclose(result[60, 180, 0], 0.2, atol=0.01)
        # Overlap region should be between 0.2 and 0.8
        overlap_val = result[60, 110, 0]
        self.assertGreater(overlap_val, 0.15)
        self.assertLess(overlap_val, 0.85)


class TestSeamFixPlanning(unittest.TestCase):
    """Seam-fix band planning and blending tests."""

    def test_seam_fix_bands_generated(self):
        tiles = plan_tiles_linear(1024, 1024, tile_size=512, tile_padding=32)
        bands = plan_seam_fix_bands(tiles, 1024, 1024, seam_fix_width=64, seam_fix_padding=16)
        self.assertGreater(len(bands), 0)

    def test_seam_fix_disabled_when_width_zero(self):
        tiles = plan_tiles_linear(1024, 1024, tile_size=512, tile_padding=32)
        bands = plan_seam_fix_bands(tiles, 1024, 1024, seam_fix_width=0)
        self.assertEqual(len(bands), 0)

    def test_seam_fix_band_within_image(self):
        tiles = plan_tiles_linear(1024, 768, tile_size=512, tile_padding=32)
        bands = plan_seam_fix_bands(tiles, 1024, 768, seam_fix_width=64, seam_fix_padding=16)
        for band in bands:
            self.assertGreaterEqual(band.band_x, 0)
            self.assertGreaterEqual(band.band_y, 0)
            self.assertLessEqual(band.band_x + band.band_w, 1024)
            self.assertLessEqual(band.band_y + band.band_h, 768)

    def test_seam_fix_has_horizontal_and_vertical(self):
        tiles = plan_tiles_linear(1024, 1024, tile_size=512, tile_padding=32)
        bands = plan_seam_fix_bands(tiles, 1024, 1024, seam_fix_width=64)
        orientations = {b.orientation for b in bands}
        # With a 2x2+ grid, we should have both
        self.assertIn("horizontal", orientations)
        self.assertIn("vertical", orientations)

    def test_seam_fix_mask_shape(self):
        band = SeamFixSpec(
            band_x=0, band_y=0, band_w=100, band_h=64,
            crop_x=0, crop_y=0, crop_w=132, crop_h=96,
            paste_x=0, paste_y=0, orientation="horizontal",
        )
        mask = make_seam_fix_mask(band, mask_blur=8)
        self.assertEqual(mask.shape, (64, 100))

    def test_seam_fix_mask_center_is_one(self):
        band = SeamFixSpec(
            band_x=0, band_y=0, band_w=100, band_h=64,
            crop_x=0, crop_y=0, crop_w=132, crop_h=96,
            paste_x=0, paste_y=0, orientation="horizontal",
        )
        mask = make_seam_fix_mask(band, mask_blur=8)
        self.assertAlmostEqual(mask[32, 50], 1.0)

    def test_paste_seam_fix_band_blends(self):
        canvas = np.ones((200, 200, 3), dtype=np.float32) * 0.5
        band = SeamFixSpec(
            band_x=50, band_y=90, band_w=100, band_h=20,
            crop_x=40, crop_y=80, crop_w=120, crop_h=40,
            paste_x=10, paste_y=10, orientation="horizontal",
        )
        band_pixels = np.ones((20, 100, 3), dtype=np.float32) * 0.9
        paste_seam_fix_band(canvas, band_pixels, band, mask_blur=4)

        # Center of the band should be close to 0.9
        center_val = canvas[100, 100, 0]
        self.assertGreater(center_val, 0.7)
        # Edges should blend toward original 0.5
        edge_val = canvas[90, 100, 0]
        self.assertLess(edge_val, 0.85)

    def test_extract_band_from_decoded(self):
        decoded = np.random.rand(96, 132, 3).astype(np.float32)
        band = SeamFixSpec(
            band_x=50, band_y=90, band_w=100, band_h=64,
            crop_x=40, crop_y=80, crop_w=132, crop_h=96,
            paste_x=10, paste_y=10, orientation="horizontal",
        )
        result = extract_band_from_decoded(decoded, band)
        self.assertEqual(result.shape, (64, 100, 3))

    def test_tile_plan_step_with_seam_fix(self):
        """TilePlanStep should output seam_fix_plan when seam_fix_width > 0."""
        step = UltimateSDUpscaleTilePlanStep()
        state = PipelineState()
        state.set("upscaled_width", 1024)
        state.set("upscaled_height", 1024)
        state.set("tile_size", 512)
        state.set("tile_padding", 32)
        state.set("traversal_mode", "linear")
        state.set("seam_fix_width", 64)
        state.set("seam_fix_padding", 16)
        state.set("seam_fix_mask_blur", 8)

        _, state = step(None, state)
        seam_plan = state.get("seam_fix_plan")
        self.assertIsNotNone(seam_plan)
        self.assertGreater(len(seam_plan), 0)

    def test_negative_seam_fix_padding_raises(self):
        step = UltimateSDUpscaleTilePlanStep()
        state = PipelineState()
        state.set("upscaled_width", 1024)
        state.set("upscaled_height", 1024)
        state.set("tile_size", 512)
        state.set("tile_padding", 32)
        state.set("traversal_mode", "linear")
        state.set("seam_fix_width", 64)
        state.set("seam_fix_padding", -5)
        state.set("seam_fix_mask_blur", 8)

        with self.assertRaises(ValueError):
            step(None, state)

    def test_negative_seam_fix_mask_blur_raises(self):
        step = UltimateSDUpscaleTilePlanStep()
        state = PipelineState()
        state.set("upscaled_width", 1024)
        state.set("upscaled_height", 1024)
        state.set("tile_size", 512)
        state.set("tile_padding", 32)
        state.set("traversal_mode", "linear")
        state.set("seam_fix_width", 64)
        state.set("seam_fix_padding", 16)
        state.set("seam_fix_mask_blur", -3)

        with self.assertRaises(ValueError):
            step(None, state)

    def test_odd_seam_fix_width_preserved(self):
        """Odd seam_fix_width should not lose a pixel."""
        tiles = plan_tiles_linear(1024, 1024, tile_size=512, tile_padding=32)
        bands_63 = plan_seam_fix_bands(tiles, 1024, 1024, seam_fix_width=63)
        # Interior bands (not clamped by image edges) should have full width
        for band in bands_63:
            if band.orientation == "horizontal":
                # Band height should be 63 for interior seams
                if band.band_y > 0 and band.band_y + band.band_h < 1024:
                    self.assertEqual(band.band_h, 63)
            else:
                if band.band_x > 0 and band.band_x + band.band_w < 1024:
                    self.assertEqual(band.band_w, 63)


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


class TestControlNetSupport(unittest.TestCase):
    def test_pipeline_inputs_include_controlnet(self):
        blocks = UltimateSDUpscaleBlocks()
        self.assertIn("control_image", blocks.input_names)
        self.assertIn("controlnet_conditioning_scale", blocks.input_names)

    def test_to_pil_rgb_image_from_chw_numpy(self):
        chw = np.zeros((3, 16, 32), dtype=np.float32)
        chw[0, :, :] = 1.0
        pil_image = _to_pil_rgb_image(chw)
        self.assertEqual(pil_image.mode, "RGB")
        self.assertEqual(pil_image.size, (32, 16))

    def test_to_pil_rgb_image_from_batched_tensor(self):
        tensor = torch.zeros((1, 3, 10, 20), dtype=torch.float32)
        tensor[:, 1, :, :] = 0.5
        pil_image = _to_pil_rgb_image(tensor)
        self.assertEqual(pil_image.mode, "RGB")
        self.assertEqual(pil_image.size, (20, 10))

    def test_tile_denoiser_selects_standard_path(self):
        class DummyDenoise:
            def __init__(self):
                self.calls = 0

            def __call__(self, components, state):
                self.calls += 1
                state.set("latents", state.get("latents"))
                return components, state

        step = UltimateSDUpscaleTileDenoiserStep()
        standard = DummyDenoise()
        controlnet = DummyDenoise()
        step._denoise = standard
        step._controlnet_denoise = controlnet

        block_state = BlockState(
            latents=torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            timesteps=torch.tensor([999], dtype=torch.float32),
            num_inference_steps=1,
            prompt_embeds=torch.zeros((1, 77, 16), dtype=torch.float32),
            pooled_prompt_embeds=torch.zeros((1, 32), dtype=torch.float32),
            add_time_ids=torch.zeros((1, 6), dtype=torch.float32),
            use_controlnet=False,
        )
        dummy_tile = TileSpec(
            core_x=0,
            core_y=0,
            core_w=8,
            core_h=8,
            crop_x=0,
            crop_y=0,
            crop_w=8,
            crop_h=8,
            paste_x=0,
            paste_y=0,
        )

        step(None, block_state, tile_idx=0, tile=dummy_tile)
        self.assertEqual(standard.calls, 1)
        self.assertEqual(controlnet.calls, 0)

    def test_tile_denoiser_selects_controlnet_path(self):
        class DummyDenoise:
            def __init__(self):
                self.calls = 0

            def __call__(self, components, state):
                self.calls += 1
                state.set("latents", state.get("latents"))
                return components, state

        step = UltimateSDUpscaleTileDenoiserStep()
        standard = DummyDenoise()
        controlnet = DummyDenoise()
        step._denoise = standard
        step._controlnet_denoise = controlnet

        block_state = BlockState(
            latents=torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            timesteps=torch.tensor([999], dtype=torch.float32),
            num_inference_steps=1,
            prompt_embeds=torch.zeros((1, 77, 16), dtype=torch.float32),
            pooled_prompt_embeds=torch.zeros((1, 32), dtype=torch.float32),
            add_time_ids=torch.zeros((1, 6), dtype=torch.float32),
            use_controlnet=True,
            controlnet_cond=torch.zeros((1, 3, 64, 64), dtype=torch.float32),
            conditioning_scale=1.0,
            controlnet_keep=[1.0],
            guess_mode=False,
        )
        dummy_tile = TileSpec(
            core_x=0,
            core_y=0,
            core_w=8,
            core_h=8,
            crop_x=0,
            crop_y=0,
            crop_w=8,
            crop_h=8,
            paste_x=0,
            paste_y=0,
        )

        step(None, block_state, tile_idx=0, tile=dummy_tile)
        self.assertEqual(standard.calls, 0)
        self.assertEqual(controlnet.calls, 1)


if __name__ == "__main__":
    unittest.main()
