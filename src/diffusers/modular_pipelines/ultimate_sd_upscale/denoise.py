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

"""Tiled img2img loop for Ultimate SD Upscale.

Architecture follows the ``LoopSequentialPipelineBlocks`` pattern used by the
SDXL denoising loop.  ``UltimateSDUpscaleTileLoopStep`` is the loop wrapper
(iterates over *tiles*); its sub-blocks are leaf blocks that handle one tile
per call:

    TilePrepareStep   – crop, VAE encode, prepare latents, tile-aware add_cond
    TileDenoiserStep  – full denoising loop (wraps ``StableDiffusionXLDenoiseStep``)
    TilePostProcessStep – decode latents, extract core, paste into canvas

SDXL blocks are reused via their public interface by creating temporary
``PipelineState`` objects, NOT by calling private helpers.
"""

import numpy as np
import PIL.Image
import torch

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import EulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from ..stable_diffusion_xl.before_denoise import (
    StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep,
    StableDiffusionXLImg2ImgPrepareLatentsStep,
)
from ..stable_diffusion_xl.decoders import StableDiffusionXLDecodeStep
from ..stable_diffusion_xl.denoise import StableDiffusionXLDenoiseStep
from ..stable_diffusion_xl.encoders import StableDiffusionXLVaeEncoderStep
from .utils_tiling import (
    SeamFixSpec,
    TileSpec,
    crop_tile,
    extract_band_from_decoded,
    extract_core_from_decoded,
    finalize_blended_canvas,
    paste_core_into_canvas,
    paste_core_into_canvas_blended,
    paste_seam_fix_band,
)


logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: populate a PipelineState from a dict
# ---------------------------------------------------------------------------

def _make_state(values: dict, kwargs_type_map: dict | None = None) -> PipelineState:
    """Create a PipelineState and set values, optionally with kwargs_type."""
    state = PipelineState()
    kwargs_type_map = kwargs_type_map or {}
    for k, v in values.items():
        state.set(k, v, kwargs_type_map.get(k))
    return state


# ---------------------------------------------------------------------------
# Loop sub-block 1: Prepare (crop + encode + timesteps + latents + add_cond)
# ---------------------------------------------------------------------------

class UltimateSDUpscaleTilePrepareStep(ModularPipelineBlocks):
    """Loop sub-block that prepares one tile for denoising.

    For each tile it:
      1. Crops the padded region from the upscaled image.
      2. Calls ``StableDiffusionXLVaeEncoderStep`` to encode to latents.
      3. Resets the scheduler step index (reuses timesteps from the outer
         set_timesteps block — does NOT re-run set_timesteps to avoid
         double-applying strength).
      4. Calls ``StableDiffusionXLImg2ImgPrepareLatentsStep``.
      5. Calls ``StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep``
         with tile-aware ``crops_coords_top_left`` and ``target_size``.

    All SDXL blocks are reused via their public ``__call__`` interface.
    """

    model_name = "stable-diffusion-xl"

    def __init__(self):
        super().__init__()
        # Store SDXL blocks as attributes (NOT in sub_blocks → remains a leaf)
        self._vae_encoder = StableDiffusionXLVaeEncoderStep()
        self._prepare_latents = StableDiffusionXLImg2ImgPrepareLatentsStep()
        self._prepare_add_cond = StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep()

    @property
    def description(self) -> str:
        return (
            "Loop sub-block: crops a tile, encodes to latents, resets scheduler "
            "timesteps, prepares latents, and computes tile-aware additional conditioning."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
            ComponentSpec("scheduler", EulerDiscreteScheduler),
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.5}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [ConfigSpec("requires_aesthetics_score", False)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("upscaled_image", type_hint=PIL.Image.Image, required=True),
            InputParam("upscaled_height", type_hint=int, required=True),
            InputParam("upscaled_width", type_hint=int, required=True),
            InputParam("generator"),
            InputParam("batch_size", type_hint=int, required=True),
            InputParam("num_images_per_prompt", type_hint=int, default=1),
            InputParam("dtype", type_hint=torch.dtype, required=True),
            InputParam("pooled_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("num_inference_steps", type_hint=int, default=50),
            InputParam("strength", type_hint=float, default=0.3),
            InputParam("timesteps", type_hint=torch.Tensor, required=True),
            InputParam("latent_timestep", type_hint=torch.Tensor, required=True),
            InputParam("denoising_start"),
            InputParam("denoising_end"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam("add_time_ids", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("negative_add_time_ids", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("timestep_cond", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, tile_idx: int, tile: TileSpec):
        # --- 1. Crop tile ---
        tile_image = crop_tile(block_state.upscaled_image, tile)

        # --- 2. VAE encode tile ---
        enc_state = _make_state({
            "image": tile_image,
            "height": tile.crop_h,
            "width": tile.crop_w,
            "generator": block_state.generator,
            "dtype": block_state.dtype,
            "preprocess_kwargs": None,
        })
        components, enc_state = self._vae_encoder(components, enc_state)
        image_latents = enc_state.get("image_latents")

        # --- 3. Reset scheduler step state for this tile ---
        # The outer set_timesteps block already computed the correct timesteps
        # and num_inference_steps (with strength applied). We must NOT re-run
        # set_timesteps here — that would double-apply strength and produce
        # 0 denoising steps. Instead, reset the scheduler's mutable step index
        # so it can iterate the same schedule again for this tile.
        scheduler = components.scheduler
        timesteps = block_state.timesteps
        num_inference_steps = block_state.num_inference_steps
        latent_timestep = block_state.latent_timestep

        # Only reset _step_index (progress counter). Do NOT touch _begin_index —
        # it holds the correct start position computed by the outer set_timesteps
        # step (e.g., step 14 for strength=0.3 with 20 steps). Resetting it to 0
        # would make the scheduler use sigmas for full noise (timestep ~999) when
        # the latents only have partial noise (timestep ~250), producing garbage.
        if hasattr(scheduler, "_step_index"):
            scheduler._step_index = None
        if hasattr(scheduler, "is_scale_input_called"):
            scheduler.is_scale_input_called = False

        # --- 4. Prepare latents ---
        lat_state = _make_state({
            "image_latents": image_latents,
            "latent_timestep": latent_timestep,
            "batch_size": block_state.batch_size,
            "num_images_per_prompt": block_state.num_images_per_prompt,
            "dtype": block_state.dtype,
            "generator": block_state.generator,
            "latents": None,
            "denoising_start": getattr(block_state, "denoising_start", None),
        })
        components, lat_state = self._prepare_latents(components, lat_state)

        # --- 5. Prepare additional conditioning (tile-aware) ---
        # crops_coords_top_left tells SDXL where this tile sits in the canvas
        # target_size is the tile's pixel dimensions
        # original_size is the full upscaled image dimensions
        cond_state = _make_state({
            "original_size": (block_state.upscaled_height, block_state.upscaled_width),
            "target_size": (tile.crop_h, tile.crop_w),
            "crops_coords_top_left": (tile.crop_y, tile.crop_x),
            "negative_original_size": None,
            "negative_target_size": None,
            "negative_crops_coords_top_left": (0, 0),
            "num_images_per_prompt": block_state.num_images_per_prompt,
            "aesthetic_score": 6.0,
            "negative_aesthetic_score": 2.0,
            "latents": lat_state.get("latents"),
            "pooled_prompt_embeds": block_state.pooled_prompt_embeds,
            "batch_size": block_state.batch_size,
        })
        components, cond_state = self._prepare_add_cond(components, cond_state)

        # --- Write results to block_state ---
        # timesteps/num_inference_steps/latent_timestep are from the outer
        # set_timesteps step (already in block_state), no need to overwrite.
        block_state.latents = lat_state.get("latents")
        block_state.add_time_ids = cond_state.get("add_time_ids")
        block_state.negative_add_time_ids = cond_state.get("negative_add_time_ids")
        block_state.timestep_cond = cond_state.get("timestep_cond")

        return components, block_state


# ---------------------------------------------------------------------------
# Loop sub-block 2: Denoise
# ---------------------------------------------------------------------------

class UltimateSDUpscaleTileDenoiserStep(ModularPipelineBlocks):
    """Loop sub-block that runs the full denoising loop for one tile.

    Wraps ``StableDiffusionXLDenoiseStep`` (itself a
    ``LoopSequentialPipelineBlocks`` over timesteps).  Stored as an attribute,
    not in ``sub_blocks``, so this block remains a leaf.
    """

    model_name = "stable-diffusion-xl"

    def __init__(self):
        super().__init__()
        self._denoise = StableDiffusionXLDenoiseStep()

    @property
    def description(self) -> str:
        return (
            "Loop sub-block: runs the SDXL denoising loop for one tile, "
            "wrapping StableDiffusionXLDenoiseStep."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec("scheduler", EulerDiscreteScheduler),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.5}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam("timesteps", type_hint=torch.Tensor, required=True),
            InputParam("num_inference_steps", type_hint=int, required=True),
            # Denoiser input fields (prompt embeddings, add_time_ids, etc.)
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam("pooled_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("negative_pooled_prompt_embeds", type_hint=torch.Tensor),
            InputParam("add_time_ids", type_hint=torch.Tensor, required=True),
            InputParam("negative_add_time_ids", type_hint=torch.Tensor),
            InputParam("timestep_cond", type_hint=torch.Tensor),
            InputParam("eta", type_hint=float, default=0.0),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="Denoised latents."),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, tile_idx: int, tile: TileSpec):
        # Build a PipelineState with all the data the SDXL denoise step needs
        denoiser_fields = {
            "prompt_embeds": block_state.prompt_embeds,
            "negative_prompt_embeds": getattr(block_state, "negative_prompt_embeds", None),
            "pooled_prompt_embeds": block_state.pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": getattr(block_state, "negative_pooled_prompt_embeds", None),
            "add_time_ids": block_state.add_time_ids,
            "negative_add_time_ids": getattr(block_state, "negative_add_time_ids", None),
        }
        # Add optional fields
        ip_embeds = getattr(block_state, "ip_adapter_embeds", None)
        neg_ip_embeds = getattr(block_state, "negative_ip_adapter_embeds", None)
        if ip_embeds is not None:
            denoiser_fields["ip_adapter_embeds"] = ip_embeds
        if neg_ip_embeds is not None:
            denoiser_fields["negative_ip_adapter_embeds"] = neg_ip_embeds

        kwargs_type_map = {k: "denoiser_input_fields" for k in denoiser_fields}

        all_values = {
            **denoiser_fields,
            "latents": block_state.latents,
            "timesteps": block_state.timesteps,
            "num_inference_steps": block_state.num_inference_steps,
            "timestep_cond": getattr(block_state, "timestep_cond", None),
            "eta": getattr(block_state, "eta", 0.0),
            "generator": getattr(block_state, "generator", None),
        }

        denoise_state = _make_state(all_values, kwargs_type_map)
        components, denoise_state = self._denoise(components, denoise_state)

        block_state.latents = denoise_state.get("latents")
        return components, block_state


# ---------------------------------------------------------------------------
# Loop sub-block 3: Decode + paste into canvas
# ---------------------------------------------------------------------------

class UltimateSDUpscaleTilePostProcessStep(ModularPipelineBlocks):
    """Loop sub-block that decodes one tile and pastes the core into the canvas.

    Supports two blending modes:
    - ``"none"``: Non-overlapping core paste (fastest, default).
    - ``"gradient"``: Gradient overlap blending for smoother tile transitions.
    """

    model_name = "stable-diffusion-xl"

    def __init__(self):
        super().__init__()
        self._decode = StableDiffusionXLDecodeStep()

    @property
    def description(self) -> str:
        return (
            "Loop sub-block: decodes latents to an image via StableDiffusionXLDecodeStep, "
            "then extracts the core region and pastes it into the output canvas. "
            "Supports 'none' and 'gradient' blending modes."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return []  # Canvas is modified in-place on block_state

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, tile_idx: int, tile: TileSpec):
        decode_state = _make_state({
            "latents": block_state.latents,
            "output_type": "np",
        })
        components, decode_state = self._decode(components, decode_state)
        decoded_images = decode_state.get("images")

        decoded_np = decoded_images[0]  # shape: (crop_h, crop_w, 3)

        if decoded_np.shape[0] != tile.crop_h or decoded_np.shape[1] != tile.crop_w:
            pil_tile = PIL.Image.fromarray((np.clip(decoded_np, 0, 1) * 255).astype(np.uint8))
            pil_tile = pil_tile.resize((tile.crop_w, tile.crop_h), PIL.Image.LANCZOS)
            decoded_np = np.array(pil_tile).astype(np.float32) / 255.0

        core = extract_core_from_decoded(decoded_np, tile)

        blend_mode = getattr(block_state, "blend_mode", "none")
        if blend_mode == "gradient":
            overlap = getattr(block_state, "gradient_blend_overlap", 0)
            paste_core_into_canvas_blended(
                block_state.canvas, block_state.weight_map, core, tile, overlap
            )
        elif blend_mode == "none":
            paste_core_into_canvas(block_state.canvas, core, tile)
        else:
            raise ValueError(
                f"Unsupported blend_mode '{blend_mode}'. "
                "Supported modes: 'none', 'gradient'."
            )

        return components, block_state


# ---------------------------------------------------------------------------
# Tile loop wrapper (LoopSequentialPipelineBlocks)
# ---------------------------------------------------------------------------

class UltimateSDUpscaleTileLoopStep(LoopSequentialPipelineBlocks):
    """Tile loop that iterates over the tile plan, running sub-blocks per tile.

    Supports:
    - Two blending modes: ``"none"`` (core paste) and ``"gradient"`` (overlap blending)
    - Optional seam-fix pass: re-denoises narrow bands along tile boundaries
      with feathered mask blending

    Sub-blocks:
        - ``UltimateSDUpscaleTilePrepareStep``    – crop, encode, prepare
        - ``UltimateSDUpscaleTileDenoiserStep``    – denoising loop
        - ``UltimateSDUpscaleTilePostProcessStep`` – decode + paste
    """

    model_name = "stable-diffusion-xl"

    block_classes = [
        UltimateSDUpscaleTilePrepareStep,
        UltimateSDUpscaleTileDenoiserStep,
        UltimateSDUpscaleTilePostProcessStep,
    ]
    block_names = ["tile_prepare", "tile_denoise", "tile_postprocess"]

    @property
    def description(self) -> str:
        return (
            "Tile loop that iterates over the tile plan and runs sub-blocks per tile.\n"
            "Supports 'none' and 'gradient' blending modes, plus optional seam-fix pass.\n"
            "Sub-blocks:\n"
            "  - UltimateSDUpscaleTilePrepareStep: crop, VAE encode, set timesteps, "
            "prepare latents, tile-aware add_cond\n"
            "  - UltimateSDUpscaleTileDenoiserStep: SDXL denoising loop\n"
            "  - UltimateSDUpscaleTilePostProcessStep: decode + paste core into canvas"
        )

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("tile_plan", type_hint=list, required=True,
                       description="List of TileSpec from the tile planning step."),
            InputParam("upscaled_image", type_hint=PIL.Image.Image, required=True),
            InputParam("upscaled_height", type_hint=int, required=True),
            InputParam("upscaled_width", type_hint=int, required=True),
            InputParam("tile_padding", type_hint=int, default=32),
            InputParam("output_type", type_hint=str, default="pil"),
            InputParam("blend_mode", type_hint=str, default="none",
                       description="Blending mode: 'none' (core paste) or 'gradient' (overlap blending)."),
            InputParam("gradient_blend_overlap", type_hint=int, default=16,
                       description="Width of gradient ramp in pixels for 'gradient' blend mode."),
            InputParam("seam_fix_plan", type_hint=list, default=[],
                       description="List of SeamFixSpec from tile planning. Empty disables seam fix."),
            InputParam("seam_fix_mask_blur", type_hint=int, default=8,
                       description="Feathering width for seam-fix band blending."),
            InputParam("seam_fix_strength", type_hint=float, default=0.3,
                       description="Denoise strength for seam-fix bands."),
        ]

    @property
    def loop_intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("images", type_hint=list, description="Final stitched output images."),
        ]

    def _run_seam_fix_band(self, components, block_state, band: SeamFixSpec, band_idx: int):
        """Re-denoise one seam-fix band and blend it into the canvas."""
        # Crop the band region directly from the float canvas to avoid
        # full-canvas uint8 quantization per band (quality + perf).
        crop_region = np.clip(
            block_state.canvas[band.crop_y:band.crop_y + band.crop_h,
                               band.crop_x:band.crop_x + band.crop_w],
            0, 1,
        )
        crop_uint8 = (crop_region * 255).astype(np.uint8)
        band_crop_pil = PIL.Image.fromarray(crop_uint8)

        # The PIL image is the crop region only, so the tile spec must use
        # 0-based coordinates (the entire image IS the crop).
        band_tile = TileSpec(
            core_x=band.paste_x, core_y=band.paste_y,
            core_w=band.band_w, core_h=band.band_h,
            crop_x=0, crop_y=0,
            crop_w=band.crop_w, crop_h=band.crop_h,
            paste_x=band.paste_x, paste_y=band.paste_y,
        )

        # Store original upscaled_image and swap in the band crop
        original_image = block_state.upscaled_image
        block_state.upscaled_image = band_crop_pil

        # Override strength for seam fix
        original_strength = block_state.strength
        block_state.strength = getattr(block_state, "seam_fix_strength", 0.3)

        # Run prepare + denoise (reuse existing sub-blocks)
        prepare_block = self.sub_blocks["tile_prepare"]
        denoise_block = self.sub_blocks["tile_denoise"]

        components, block_state = prepare_block(components, block_state, tile_idx=band_idx, tile=band_tile)
        components, block_state = denoise_block(components, block_state, tile_idx=band_idx, tile=band_tile)

        # Decode the band
        decode_state = _make_state({
            "latents": block_state.latents,
            "output_type": "np",
        })
        decode_block = self.sub_blocks["tile_postprocess"]._decode
        components, decode_state = decode_block(components, decode_state)
        decoded_np = decode_state.get("images")[0]

        if decoded_np.shape[0] != band.crop_h or decoded_np.shape[1] != band.crop_w:
            pil_band = PIL.Image.fromarray((np.clip(decoded_np, 0, 1) * 255).astype(np.uint8))
            pil_band = pil_band.resize((band.crop_w, band.crop_h), PIL.Image.LANCZOS)
            decoded_np = np.array(pil_band).astype(np.float32) / 255.0

        # Extract and paste band with feathered mask
        band_pixels = extract_band_from_decoded(decoded_np, band)
        seam_fix_mask_blur = getattr(block_state, "seam_fix_mask_blur", 8)
        paste_seam_fix_band(block_state.canvas, band_pixels, band, seam_fix_mask_blur)

        # Restore original values
        block_state.upscaled_image = original_image
        block_state.strength = original_strength

        return components, block_state

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        tile_plan = block_state.tile_plan
        h = block_state.upscaled_height
        w = block_state.upscaled_width
        output_type = block_state.output_type
        blend_mode = getattr(block_state, "blend_mode", "none")
        if blend_mode not in ("none", "gradient"):
            raise ValueError(
                f"Unsupported blend_mode '{blend_mode}'. Supported: 'none', 'gradient'."
            )

        # Initialize canvas
        block_state.canvas = np.zeros((h, w, 3), dtype=np.float32)
        if blend_mode == "gradient":
            block_state.weight_map = np.zeros((h, w), dtype=np.float32)
            block_state.blend_mode = blend_mode
            block_state.gradient_blend_overlap = getattr(block_state, "gradient_blend_overlap", 16)

        num_tiles = len(tile_plan)
        seam_fix_plan = getattr(block_state, "seam_fix_plan", []) or []
        total_steps = num_tiles + len(seam_fix_plan)

        logger.info(
            f"Processing {num_tiles} tiles"
            + (f" (blend_mode={blend_mode})" if blend_mode != "none" else "")
            + (f" + {len(seam_fix_plan)} seam-fix bands" if seam_fix_plan else "")
        )

        with self.progress_bar(total=total_steps) as progress_bar:
            # Main tile loop
            for i, tile in enumerate(tile_plan):
                logger.debug(
                    f"Tile {i + 1}/{num_tiles}: core=({tile.core_x},{tile.core_y},{tile.core_w},{tile.core_h}) "
                    f"crop=({tile.crop_x},{tile.crop_y},{tile.crop_w},{tile.crop_h})"
                )
                components, block_state = self.loop_step(components, block_state, tile_idx=i, tile=tile)
                progress_bar.update()

            # Finalize gradient blending before seam fix
            if blend_mode == "gradient":
                block_state.canvas = finalize_blended_canvas(block_state.canvas, block_state.weight_map)

            # Seam-fix pass
            for j, band in enumerate(seam_fix_plan):
                logger.debug(
                    f"Seam-fix {j + 1}/{len(seam_fix_plan)}: "
                    f"band=({band.band_x},{band.band_y},{band.band_w},{band.band_h}) "
                    f"{band.orientation}"
                )
                components, block_state = self._run_seam_fix_band(components, block_state, band, j)
                progress_bar.update()

        # Finalize output
        result = np.clip(block_state.canvas, 0.0, 1.0)
        result_uint8 = (result * 255).astype(np.uint8)

        if output_type == "pil":
            block_state.images = [PIL.Image.fromarray(result_uint8)]
        elif output_type == "np":
            block_state.images = [result]
        elif output_type == "pt":
            block_state.images = [torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0)]
        else:
            block_state.images = [PIL.Image.fromarray(result_uint8)]

        self.set_block_state(state, block_state)
        return components, state
