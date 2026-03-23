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

"""Top-level block composition for Ultimate SD Upscale.

The pipeline preserves the standard SDXL img2img block graph as closely as
possible, inserting upscale and tile-plan steps and wrapping the per-tile
work in a ``LoopSequentialPipelineBlocks``::

    text_encoder → upscale → tile_plan → input → set_timesteps → tiled_img2img

Inside ``tiled_img2img`` (tile loop), each tile runs:

    tile_prepare → tile_denoise → tile_postprocess

The standard ``input`` and ``set_timesteps`` blocks run once (outside the
loop) to establish shared batch/dtype/timestep defaults. The tile-prepare
sub-block also performs a per-tile set-timesteps call to reset mutable
scheduler step state safely before each denoise run.
"""

from ...utils import logging
from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from ..stable_diffusion_xl.before_denoise import (
    StableDiffusionXLImg2ImgSetTimestepsStep,
    StableDiffusionXLInputStep,
)
from ..stable_diffusion_xl.encoders import StableDiffusionXLTextEncoderStep
from .denoise import UltimateSDUpscaleTileLoopStep
from .input import UltimateSDUpscaleTilePlanStep, UltimateSDUpscaleUpscaleStep


logger = logging.get_logger(__name__)


class UltimateSDUpscaleBlocks(SequentialPipelineBlocks):
    """Modular pipeline blocks for Ultimate SD Upscale (SDXL).

    Block graph::

        [0] text_encoder   – StableDiffusionXLTextEncoderStep (reused)
        [1] upscale        – UltimateSDUpscaleUpscaleStep (new)
        [2] tile_plan      – UltimateSDUpscaleTilePlanStep (new)
        [3] input          – StableDiffusionXLInputStep (reused)
        [4] set_timesteps  – StableDiffusionXLImg2ImgSetTimestepsStep (reused)
        [5] tiled_img2img  – UltimateSDUpscaleTileLoopStep (new tile loop)

    Pass 1 scope:
        - SDXL only, Lanczos upscale, linear traversal
        - Non-overlapping core paste (padded crop for denoise context only)
        - Tile-aware SDXL conditioning (crops_coords_top_left per tile)
        - No seam fixing yet
    """

    block_classes = [
        StableDiffusionXLTextEncoderStep,
        UltimateSDUpscaleUpscaleStep,
        UltimateSDUpscaleTilePlanStep,
        StableDiffusionXLInputStep,
        StableDiffusionXLImg2ImgSetTimestepsStep,
        UltimateSDUpscaleTileLoopStep,
    ]
    block_names = [
        "text_encoder",
        "upscale",
        "tile_plan",
        "input",
        "set_timesteps",
        "tiled_img2img",
    ]

    _workflow_map = {
        "upscale": {"image": True, "prompt": True},
    }

    @property
    def description(self):
        return (
            "Modular pipeline for Ultimate SD Upscale using Stable Diffusion XL.\n"
            "Upscales an input image and refines it tile-by-tile using img2img "
            "denoising with configurable tile size, overlap padding, and strength.\n"
            "Pass 1: Lanczos upscale, linear traversal, non-overlapping core paste."
        )

    @property
    def outputs(self):
        return [OutputParam.template("images")]
