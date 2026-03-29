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
from tqdm.auto import tqdm

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from ...schedulers import EulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from ..stable_diffusion_xl.before_denoise import (
    StableDiffusionXLControlNetInputStep,
    StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep,
    StableDiffusionXLImg2ImgPrepareLatentsStep,
    prepare_latents_img2img,
)
from ..stable_diffusion_xl.decoders import StableDiffusionXLDecodeStep
from ..stable_diffusion_xl.denoise import StableDiffusionXLControlNetDenoiseStep, StableDiffusionXLDenoiseStep
from ..stable_diffusion_xl.encoders import StableDiffusionXLVaeEncoderStep
from .utils_tiling import (
    LatentTileSpec,
    SeamFixSpec,
    TileSpec,
    crop_tile,
    extract_band_from_decoded,
    extract_core_from_decoded,
    finalize_blended_canvas,
    paste_core_into_canvas,
    paste_core_into_canvas_blended,
    paste_seam_fix_band,
    plan_latent_tiles,
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


def _to_pil_rgb_image(image) -> PIL.Image.Image:
    """Convert a tensor/ndarray/PIL image to a RGB PIL image."""
    if isinstance(image, PIL.Image.Image):
        return image.convert("RGB")

    if torch.is_tensor(image):
        tensor = image.detach().cpu()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"`control_image` tensor batch must be 1 for tiled upscaling, got shape {tuple(tensor.shape)}."
                )
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        image = tensor.numpy()

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise ValueError(
                    f"`control_image` ndarray batch must be 1 for tiled upscaling, got shape {array.shape}."
                )
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.ndim != 3:
            raise ValueError(f"`control_image` must have 2 or 3 dimensions, got shape {array.shape}.")
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.shape[-1] != 3:
            raise ValueError(f"`control_image` channel dimension must be 1/3/4, got shape {array.shape}.")
        if array.dtype != np.uint8:
            array = np.asarray(array, dtype=np.float32)
            max_val = float(np.max(array)) if array.size > 0 else 1.0
            if max_val <= 1.0:
                array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        return PIL.Image.fromarray(array).convert("RGB")

    raise ValueError(
        f"Unsupported `control_image` type {type(image)}. Expected PIL.Image, torch.Tensor, or numpy.ndarray."
    )


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
        self._prepare_controlnet = StableDiffusionXLControlNetInputStep()

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
            ComponentSpec(
                "control_image_processor",
                VaeImageProcessor,
                config=FrozenDict({"do_convert_rgb": True, "do_normalize": False}),
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
            InputParam("use_controlnet", type_hint=bool, default=False),
            InputParam("control_image_processed"),
            InputParam("control_guidance_start", default=0.0),
            InputParam("control_guidance_end", default=1.0),
            InputParam("controlnet_conditioning_scale", default=1.0),
            InputParam("guess_mode", default=False),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam("add_time_ids", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("negative_add_time_ids", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("timestep_cond", type_hint=torch.Tensor),
            OutputParam("controlnet_cond", type_hint=torch.Tensor),
            OutputParam("conditioning_scale"),
            OutputParam("controlnet_keep", type_hint=list[float]),
            OutputParam("guess_mode", type_hint=bool),
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
        # Build clean init latents first (no random noise yet), then add tile noise.
        # Using a global noise map keeps noise spatially consistent across tiles and
        # greatly reduces cross-tile drift/artifacts.
        clean_latents = prepare_latents_img2img(
            components.vae,
            components.scheduler,
            image_latents,
            latent_timestep,
            block_state.batch_size,
            block_state.num_images_per_prompt,
            block_state.dtype,
            image_latents.device,
            generator=None,
            add_noise=False,
        )

        latent_h, latent_w = clean_latents.shape[-2], clean_latents.shape[-1]
        global_noise_map = getattr(block_state, "global_noise_map", None)
        if global_noise_map is not None:
            vae_scale_factor = int(getattr(block_state, "global_noise_scale", 8))
            y0 = max(0, tile.crop_y // vae_scale_factor)
            x0 = max(0, tile.crop_x // vae_scale_factor)
            max_y0 = max(0, global_noise_map.shape[-2] - latent_h)
            max_x0 = max(0, global_noise_map.shape[-1] - latent_w)
            y0 = min(y0, max_y0)
            x0 = min(x0, max_x0)
            tile_noise = global_noise_map[:, :, y0 : y0 + latent_h, x0 : x0 + latent_w]

            # Defensive fallback if latent shape and crop math ever diverge.
            if tile_noise.shape != clean_latents.shape:
                tile_noise = randn_tensor(
                    clean_latents.shape,
                    generator=block_state.generator,
                    device=clean_latents.device,
                    dtype=clean_latents.dtype,
                )
        else:
            tile_noise = randn_tensor(
                clean_latents.shape,
                generator=block_state.generator,
                device=clean_latents.device,
                dtype=clean_latents.dtype,
            )

        pre_noised_latents = components.scheduler.add_noise(clean_latents, tile_noise, latent_timestep)

        lat_state = _make_state({
            "image_latents": image_latents,
            "latent_timestep": latent_timestep,
            "batch_size": block_state.batch_size,
            "num_images_per_prompt": block_state.num_images_per_prompt,
            "dtype": block_state.dtype,
            "generator": block_state.generator,
            "latents": pre_noised_latents,
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
        if getattr(block_state, "use_controlnet", False):
            control_tile = crop_tile(block_state.control_image_processed, tile)
            control_state = _make_state({
                "control_image": control_tile,
                "control_guidance_start": getattr(block_state, "control_guidance_start", 0.0),
                "control_guidance_end": getattr(block_state, "control_guidance_end", 1.0),
                "controlnet_conditioning_scale": getattr(block_state, "controlnet_conditioning_scale", 1.0),
                "guess_mode": getattr(block_state, "guess_mode", False),
                "num_images_per_prompt": block_state.num_images_per_prompt,
                "latents": block_state.latents,
                "batch_size": block_state.batch_size,
                "timesteps": block_state.timesteps,
                "crops_coords": None,
            })
            components, control_state = self._prepare_controlnet(components, control_state)
            block_state.controlnet_cond = control_state.get("controlnet_cond")
            block_state.conditioning_scale = control_state.get("conditioning_scale")
            block_state.controlnet_keep = control_state.get("controlnet_keep")
            block_state.guess_mode = control_state.get("guess_mode")
        else:
            block_state.controlnet_cond = None
            block_state.conditioning_scale = None
            block_state.controlnet_keep = None

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
        self._controlnet_denoise = StableDiffusionXLControlNetDenoiseStep()

    @property
    def description(self) -> str:
        return (
            "Loop sub-block: runs the SDXL denoising loop for one tile, "
            "with optional ControlNet conditioning."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec("scheduler", EulerDiscreteScheduler),
            ComponentSpec("controlnet", ControlNetModel),
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
            InputParam("use_controlnet", type_hint=bool, default=False),
            InputParam("controlnet_cond", type_hint=torch.Tensor),
            InputParam("conditioning_scale"),
            InputParam("controlnet_keep", type_hint=list[float]),
            InputParam("guess_mode", type_hint=bool, default=False),
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
        use_controlnet = bool(getattr(block_state, "use_controlnet", False))
        if use_controlnet:
            all_values.update(
                {
                    "controlnet_cond": block_state.controlnet_cond,
                    "conditioning_scale": block_state.conditioning_scale,
                    "guess_mode": getattr(block_state, "guess_mode", False),
                    "controlnet_keep": block_state.controlnet_keep,
                    "controlnet_kwargs": getattr(block_state, "controlnet_kwargs", {}),
                }
            )

        denoise_state = _make_state(all_values, kwargs_type_map)
        if use_controlnet:
            components, denoise_state = self._controlnet_denoise(components, denoise_state)
        else:
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
            InputParam("control_image",
                       description="Optional ControlNet conditioning image. If provided, tile denoising uses ControlNet."),
            InputParam("control_guidance_start", default=0.0),
            InputParam("control_guidance_end", default=1.0),
            InputParam("controlnet_conditioning_scale", default=1.0),
            InputParam("guess_mode", default=False),
            InputParam("guidance_scale", type_hint=float, default=7.5,
                       description="Classifier-Free Guidance scale. Higher values produce images more aligned "
                       "with the prompt at the expense of lower image quality."),
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
        original_control_image = getattr(block_state, "control_image_processed", None)
        if getattr(block_state, "use_controlnet", False) and original_control_image is not None:
            block_state.control_image_processed = original_control_image.crop(
                (band.crop_x, band.crop_y, band.crop_x + band.crop_w, band.crop_y + band.crop_h)
            )

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
        if getattr(block_state, "use_controlnet", False):
            block_state.control_image_processed = original_control_image
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

        # --- Configure guidance_scale on guider ---
        guidance_scale = getattr(block_state, "guidance_scale", 7.5)
        components.guider.guidance_scale = guidance_scale

        control_image = getattr(block_state, "control_image", None)
        block_state.use_controlnet = control_image is not None
        if block_state.use_controlnet:
            if isinstance(control_image, list):
                raise ValueError(
                    "Ultimate SD Upscale currently supports a single `control_image`, not a list."
                )
            if not hasattr(components, "controlnet") or components.controlnet is None:
                raise ValueError(
                    "`control_image` was provided but `controlnet` component is missing. "
                    "Load a ControlNet model (for example, a tile model) into `pipe.controlnet`."
                )
            block_state.control_image_processed = _to_pil_rgb_image(control_image)
            if block_state.control_image_processed.size != (w, h):
                block_state.control_image_processed = block_state.control_image_processed.resize((w, h), PIL.Image.LANCZOS)
            logger.info("ControlNet conditioning enabled for tiled denoising.")

        # Enable VAE tiling for memory-efficient encode/decode of large images.
        # This lets the UNet process the full image (no tile seams) while the
        # VAE handles memory via its own internal tiling.
        if hasattr(components.vae, "enable_tiling"):
            components.vae.enable_tiling()

        # Initialize canvas
        block_state.canvas = np.zeros((h, w, 3), dtype=np.float32)

        # Prepare one global latent noise tensor and crop from it per tile.
        # This keeps stochasticity consistent across tile boundaries.
        vae_scale_factor = int(getattr(components, "vae_scale_factor", 8))
        latent_h = max(1, h // vae_scale_factor)
        latent_w = max(1, w // vae_scale_factor)
        effective_batch = block_state.batch_size * block_state.num_images_per_prompt
        block_state.global_noise_map = randn_tensor(
            (effective_batch, 4, latent_h, latent_w),
            generator=getattr(block_state, "generator", None),
            device=components._execution_device,
            dtype=block_state.dtype,
        )
        block_state.global_noise_scale = vae_scale_factor

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


# =============================================================================
# MultiDiffusion: latent-space noise prediction blending
# =============================================================================


def _make_cosine_tile_weight(
    h: int, w: int, overlap: int, device, dtype,
    is_top: bool = False, is_bottom: bool = False,
    is_left: bool = False, is_right: bool = False,
) -> torch.Tensor:
    """Create a boundary-aware 2D cosine-ramp weight for MultiDiffusion blending.

    Weight is 1.0 in the center and smoothly fades at edges that overlap with
    neighboring tiles. Edges that touch the image boundary keep weight=1.0 to
    prevent noise amplification from dividing by near-zero weights.

    Args:
        h: Tile height in latent pixels.
        w: Tile width in latent pixels.
        overlap: Overlap in latent pixels.
        device: Torch device.
        dtype: Torch dtype.
        is_top: True if this tile touches the top image boundary.
        is_bottom: True if this tile touches the bottom image boundary.
        is_left: True if this tile touches the left image boundary.
        is_right: True if this tile touches the right image boundary.

    Returns:
        Tensor of shape ``(1, 1, h, w)`` for broadcasting.
    """
    import math

    def _ramp(length, overlap_size, keep_start, keep_end):
        ramp = torch.ones(length, device=device, dtype=dtype)
        if overlap_size > 0 and length > 2 * overlap_size:
            fade = 0.5 * (1.0 - torch.cos(torch.linspace(0, math.pi, overlap_size, device=device, dtype=dtype)))
            if not keep_start:
                ramp[:overlap_size] = fade
            if not keep_end:
                ramp[-overlap_size:] = fade.flip(0)
        return ramp

    w_h = _ramp(h, overlap, keep_start=is_top, keep_end=is_bottom)
    w_w = _ramp(w, overlap, keep_start=is_left, keep_end=is_right)
    return (w_h[:, None] * w_w[None, :]).unsqueeze(0).unsqueeze(0)


class UltimateSDUpscaleMultiDiffusionStep(ModularPipelineBlocks):
    """Single block that encodes, denoises with MultiDiffusion, and decodes.

    MultiDiffusion inverts the standard tile loop: the **outer** loop iterates
    over timesteps and the **inner** loop iterates over overlapping latent
    tiles. At each timestep, per-tile noise predictions are blended with
    cosine-ramp overlap weights, then a single scheduler step is applied to
    the full latent tensor. This eliminates tile-boundary artifacts because
    blending happens in noise-prediction space, not pixel space.

    The full flow:
        1. Enable VAE tiling for memory-efficient encode/decode.
        2. VAE-encode the upscaled image to full-resolution latents.
        3. Add noise at the strength-determined level.
        4. For each timestep:
            a. Plan overlapping tiles in latent space.
            b. For each tile: crop latents, run UNet (+ optional ControlNet)
               through the guider for CFG, accumulate weighted noise predictions.
            c. Normalize predictions by accumulated weights.
            d. One ``scheduler.step`` on the full blended prediction.
        5. VAE-decode the final latents.
    """

    model_name = "stable-diffusion-xl"

    def __init__(self):
        super().__init__()
        self._vae_encoder = StableDiffusionXLVaeEncoderStep()
        self._prepare_latents = StableDiffusionXLImg2ImgPrepareLatentsStep()
        self._prepare_add_cond = StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep()
        self._prepare_controlnet = StableDiffusionXLControlNetInputStep()
        self._decode = StableDiffusionXLDecodeStep()

    @property
    def description(self) -> str:
        return (
            "MultiDiffusion tiled denoising: encodes the full upscaled image, "
            "denoises with latent-space noise-prediction blending across "
            "overlapping tiles, then decodes. Produces seamless output at any "
            "resolution without tile-boundary artifacts."
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
            ComponentSpec(
                "control_image_processor",
                VaeImageProcessor,
                config=FrozenDict({"do_convert_rgb": True, "do_normalize": False}),
                default_creation_method="from_config",
            ),
            ComponentSpec("scheduler", EulerDiscreteScheduler),
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec("controlnet", ControlNetModel),
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
            InputParam("output_type", type_hint=str, default="pil"),
            # Prompt embeddings for guider
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam("negative_pooled_prompt_embeds", type_hint=torch.Tensor),
            InputParam("add_time_ids", type_hint=torch.Tensor),
            InputParam("negative_add_time_ids", type_hint=torch.Tensor),
            InputParam("eta", type_hint=float, default=0.0),
            # Guidance scale for CFG
            InputParam("guidance_scale", type_hint=float, default=7.5,
                       description="Classifier-Free Guidance scale. Higher values produce images more aligned "
                       "with the prompt at the expense of lower image quality."),
            # MultiDiffusion params
            InputParam("latent_tile_size", type_hint=int, default=64,
                       description="Tile size in latent pixels (64 = 512px). For single pass, set >= latent dims."),
            InputParam("latent_overlap", type_hint=int, default=16,
                       description="Overlap in latent pixels (16 = 128px)."),
            # ControlNet params
            InputParam("control_image",
                       description="Optional ControlNet conditioning image."),
            InputParam("control_guidance_start", default=0.0),
            InputParam("control_guidance_end", default=1.0),
            InputParam("controlnet_conditioning_scale", default=1.0),
            InputParam("guess_mode", default=False),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("images", type_hint=list, description="Final upscaled output images."),
        ]

    def _run_tile_unet(
        self,
        components,
        tile_latents: torch.Tensor,
        t: int,
        i: int,
        block_state,
        controlnet_cond_tile=None,
    ) -> torch.Tensor:
        """Run guider + UNet (+ optional ControlNet) on one tile, return noise_pred."""
        import inspect

        # Scale input
        scaled_latents = components.scheduler.scale_model_input(tile_latents, t)

        # Guider inputs
        guider_inputs = {
            "prompt_embeds": (
                getattr(block_state, "prompt_embeds", None),
                getattr(block_state, "negative_prompt_embeds", None),
            ),
            "time_ids": (
                getattr(block_state, "add_time_ids", None),
                getattr(block_state, "negative_add_time_ids", None),
            ),
            "text_embeds": (
                getattr(block_state, "pooled_prompt_embeds", None),
                getattr(block_state, "negative_pooled_prompt_embeds", None),
            ),
        }

        components.guider.set_state(
            step=i,
            num_inference_steps=block_state.num_inference_steps,
            timestep=t,
        )
        guider_state = components.guider.prepare_inputs(guider_inputs)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.unet)

            added_cond_kwargs = {
                "text_embeds": guider_state_batch.text_embeds,
                "time_ids": guider_state_batch.time_ids,
            }

            down_block_res_samples = None
            mid_block_res_sample = None

            # ControlNet forward pass
            if controlnet_cond_tile is not None and components.controlnet is not None:
                cn_added_cond = {
                    "text_embeds": guider_state_batch.text_embeds,
                    "time_ids": guider_state_batch.time_ids,
                }
                cond_scale = block_state._cn_cond_scale
                if isinstance(block_state._cn_controlnet_keep, list) and i < len(block_state._cn_controlnet_keep):
                    keep_val = block_state._cn_controlnet_keep[i]
                else:
                    keep_val = 1.0
                if isinstance(cond_scale, list):
                    cond_scale = [c * keep_val for c in cond_scale]
                else:
                    cond_scale = cond_scale * keep_val

                guess_mode = getattr(block_state, "guess_mode", False)
                if guess_mode and not components.guider.is_conditional:
                    down_block_res_samples = [torch.zeros_like(s) for s in block_state._cn_zeros_down] if hasattr(block_state, "_cn_zeros_down") else None
                    mid_block_res_sample = torch.zeros_like(block_state._cn_zeros_mid) if hasattr(block_state, "_cn_zeros_mid") else None
                else:
                    down_block_res_samples, mid_block_res_sample = components.controlnet(
                        scaled_latents,
                        t,
                        encoder_hidden_states=guider_state_batch.prompt_embeds,
                        controlnet_cond=controlnet_cond_tile,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        added_cond_kwargs=cn_added_cond,
                        return_dict=False,
                    )
                    if not hasattr(block_state, "_cn_zeros_down"):
                        block_state._cn_zeros_down = [torch.zeros_like(d) for d in down_block_res_samples]
                        block_state._cn_zeros_mid = torch.zeros_like(mid_block_res_sample)

            unet_kwargs = {
                "sample": scaled_latents,
                "timestep": t,
                "encoder_hidden_states": guider_state_batch.prompt_embeds,
                "added_cond_kwargs": added_cond_kwargs,
                "return_dict": False,
            }
            if down_block_res_samples is not None:
                unet_kwargs["down_block_additional_residuals"] = down_block_res_samples
                unet_kwargs["mid_block_additional_residual"] = mid_block_res_sample

            guider_state_batch.noise_pred = components.unet(**unet_kwargs)[0]
            components.guider.cleanup_models(components.unet)

        noise_pred = components.guider(guider_state)[0]
        return noise_pred

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        h = block_state.upscaled_height
        w = block_state.upscaled_width
        output_type = block_state.output_type
        latent_tile_size = block_state.latent_tile_size
        latent_overlap = block_state.latent_overlap

        # --- Enable VAE tiling ---
        if hasattr(components.vae, "enable_tiling"):
            components.vae.enable_tiling()

        # --- ControlNet setup ---
        control_image = getattr(block_state, "control_image", None)
        use_controlnet = False
        full_controlnet_cond = None
        ctrl_pil = None
        if control_image is not None:
            if isinstance(control_image, list):
                raise ValueError(
                    "MultiDiffusion currently supports a single `control_image`, not a list."
                )
            if not hasattr(components, "controlnet") or components.controlnet is None:
                raise ValueError(
                    "`control_image` was provided but `controlnet` component is missing. "
                    "Load a ControlNet model into `pipe.controlnet` first."
                )
            ctrl_pil = _to_pil_rgb_image(control_image)
            if ctrl_pil.size != (w, h):
                ctrl_pil = ctrl_pil.resize((w, h), PIL.Image.LANCZOS)
            use_controlnet = True

        # --- VAE encode full upscaled image ---
        enc_state = _make_state({
            "image": block_state.upscaled_image,
            "height": h,
            "width": w,
            "generator": block_state.generator,
            "dtype": block_state.dtype,
            "preprocess_kwargs": None,
        })
        components, enc_state = self._vae_encoder(components, enc_state)
        image_latents = enc_state.get("image_latents")

        # --- Prepare latents (add noise) ---
        latent_timestep = block_state.latent_timestep
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
        latents = lat_state.get("latents")

        # Prepare full-resolution ControlNet conditioning with the real latent shape.
        if use_controlnet:
            ctrl_state = _make_state({
                "control_image": ctrl_pil,
                "control_guidance_start": getattr(block_state, "control_guidance_start", 0.0),
                "control_guidance_end": getattr(block_state, "control_guidance_end", 1.0),
                "controlnet_conditioning_scale": getattr(block_state, "controlnet_conditioning_scale", 1.0),
                "guess_mode": getattr(block_state, "guess_mode", False),
                "num_images_per_prompt": block_state.num_images_per_prompt,
                "latents": latents,
                "batch_size": block_state.batch_size,
                "timesteps": block_state.timesteps,
                "crops_coords": None,
            })
            components, ctrl_state = self._prepare_controlnet(components, ctrl_state)
            full_controlnet_cond = ctrl_state.get("controlnet_cond")
            block_state._cn_cond_scale = ctrl_state.get("conditioning_scale")
            block_state._cn_controlnet_keep = ctrl_state.get("controlnet_keep")
            block_state.guess_mode = ctrl_state.get("guess_mode")
            logger.info("MultiDiffusion: ControlNet enabled.")

        # --- Prepare additional conditioning ---
        cond_state = _make_state({
            "original_size": (h, w),
            "target_size": (h, w),
            "crops_coords_top_left": (0, 0),
            "negative_original_size": None,
            "negative_target_size": None,
            "negative_crops_coords_top_left": (0, 0),
            "num_images_per_prompt": block_state.num_images_per_prompt,
            "aesthetic_score": 6.0,
            "negative_aesthetic_score": 2.0,
            "latents": latents,
            "pooled_prompt_embeds": block_state.pooled_prompt_embeds,
            "batch_size": block_state.batch_size,
        })
        components, cond_state = self._prepare_add_cond(components, cond_state)
        block_state.add_time_ids = cond_state.get("add_time_ids")
        block_state.negative_add_time_ids = cond_state.get("negative_add_time_ids")

        # --- Plan latent tiles ---
        latent_h, latent_w = latents.shape[-2], latents.shape[-1]
        tile_specs = plan_latent_tiles(latent_h, latent_w, latent_tile_size, latent_overlap)
        num_tiles = len(tile_specs)
        logger.info(f"MultiDiffusion: {num_tiles} latent tiles ({latent_h}x{latent_w}, tile={latent_tile_size}, overlap={latent_overlap})")

        # --- Configure guidance_scale on guider ---
        guidance_scale = getattr(block_state, "guidance_scale", 7.5)
        components.guider.guidance_scale = guidance_scale

        # --- Guider setup ---
        disable_guidance = True if components.unet.config.time_cond_proj_dim is not None else False
        if disable_guidance:
            components.guider.disable()
        else:
            components.guider.enable()

        # --- MultiDiffusion denoise loop ---
        timesteps = block_state.timesteps
        vae_scale_factor = int(getattr(components, "vae_scale_factor", 8))
        progress_kwargs = getattr(components, "_progress_bar_config", {})
        if not isinstance(progress_kwargs, dict):
            progress_kwargs = {}

        for i, t in enumerate(
            tqdm(timesteps, total=block_state.num_inference_steps, desc="MultiDiffusion", **progress_kwargs)
        ):
            # Accumulators for noise prediction blending
            # Use float32 accumulators for numerical stability; fp16 can underflow
            # tiny overlap weights and produce NaNs in normalization.
            noise_pred_accum = torch.zeros_like(latents, dtype=torch.float32)
            weight_accum = torch.zeros(
                1, 1, latent_h, latent_w,
                device=latents.device, dtype=torch.float32,
            )

            for tile in tile_specs:
                # Crop tile from full latents
                tile_latents = latents[:, :, tile.y:tile.y + tile.h, tile.x:tile.x + tile.w].clone()

                # Crop ControlNet cond for this tile (pixel space)
                cn_tile = None
                if use_controlnet and full_controlnet_cond is not None:
                    py = tile.y * vae_scale_factor
                    px = tile.x * vae_scale_factor
                    ph = tile.h * vae_scale_factor
                    pw = tile.w * vae_scale_factor
                    cn_tile = full_controlnet_cond[:, :, py:py + ph, px:px + pw]

                # Run UNet (+ ControlNet + guider) on this tile
                tile_noise_pred = self._run_tile_unet(
                    components, tile_latents, t, i, block_state, cn_tile,
                )

                # Boundary-aware cosine weight for this tile
                tile_weight = _make_cosine_tile_weight(
                    tile.h, tile.w, latent_overlap,
                    latents.device, torch.float32,
                    is_top=(tile.y == 0),
                    is_bottom=(tile.y + tile.h >= latent_h),
                    is_left=(tile.x == 0),
                    is_right=(tile.x + tile.w >= latent_w),
                )

                # Accumulate
                noise_pred_accum[:, :, tile.y:tile.y + tile.h, tile.x:tile.x + tile.w] += (
                    tile_noise_pred.to(torch.float32) * tile_weight
                )
                weight_accum[:, :, tile.y:tile.y + tile.h, tile.x:tile.x + tile.w] += tile_weight

            # Normalize (MultiDiffusion weighted average)
            blended_noise_pred = noise_pred_accum / weight_accum.clamp(min=1e-6)
            blended_noise_pred = torch.nan_to_num(blended_noise_pred, nan=0.0, posinf=0.0, neginf=0.0)
            blended_noise_pred = blended_noise_pred.to(latents.dtype)

            # Scheduler step on full latent
            latents_dtype = latents.dtype
            latents = components.scheduler.step(
                blended_noise_pred, t, latents,
                return_dict=False,
            )[0]
            if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)

        # --- Decode ---
        decode_state = _make_state({
            "latents": latents,
            "output_type": "np",
        })
        components, decode_state = self._decode(components, decode_state)
        decoded_images = decode_state.get("images")
        decoded_np = decoded_images[0]

        # Resize if VAE output doesn't exactly match target
        if decoded_np.shape[0] != h or decoded_np.shape[1] != w:
            pil_out = PIL.Image.fromarray((np.clip(decoded_np, 0, 1) * 255).astype(np.uint8))
            pil_out = pil_out.resize((w, h), PIL.Image.LANCZOS)
            decoded_np = np.array(pil_out).astype(np.float32) / 255.0

        # Format output
        result_uint8 = (np.clip(decoded_np, 0, 1) * 255).astype(np.uint8)
        if output_type == "pil":
            block_state.images = [PIL.Image.fromarray(result_uint8)]
        elif output_type == "np":
            block_state.images = [decoded_np]
        elif output_type == "pt":
            block_state.images = [torch.from_numpy(decoded_np).permute(2, 0, 1).unsqueeze(0)]
        else:
            block_state.images = [PIL.Image.fromarray(result_uint8)]

        self.set_block_state(state, block_state)
        return components, state
