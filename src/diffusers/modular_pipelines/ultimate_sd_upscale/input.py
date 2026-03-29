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

import PIL.Image
import torch

from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import InputParam, OutputParam
from .utils_tiling import plan_seam_fix_bands, plan_tiles_chess, plan_tiles_linear, validate_tile_params


logger = logging.get_logger(__name__)


class UltimateSDUpscaleUpscaleStep(ModularPipelineBlocks):
    """Upscales the input image using Lanczos interpolation.

    This is the first custom step in the Ultimate SD Upscale workflow.
    It takes an input image and upscale factor, producing an upscaled image
    that subsequent tile steps will refine.
    """

    @property
    def description(self) -> str:
        return (
            "Upscale step that resizes the input image by a given factor.\n"
            "Currently supports Lanczos interpolation. Model-based upscalers "
            "can be added in future passes."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "image",
                type_hint=PIL.Image.Image,
                required=True,
                description="The input image to upscale and refine.",
            ),
            InputParam(
                "upscale_factor",
                type_hint=float,
                default=2.0,
                description="Factor by which to upscale the input image.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "upscaled_image",
                type_hint=PIL.Image.Image,
                description="The upscaled image before tile-based refinement.",
            ),
            OutputParam(
                "upscaled_width",
                type_hint=int,
                description="Width of the upscaled image.",
            ),
            OutputParam(
                "upscaled_height",
                type_hint=int,
                description="Height of the upscaled image.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        image = block_state.image
        upscale_factor = block_state.upscale_factor

        if not isinstance(image, PIL.Image.Image):
            raise ValueError(
                f"Expected `image` to be a PIL.Image.Image, got {type(image)}. "
                "Please pass a PIL image to the pipeline."
            )

        new_width = int(image.width * upscale_factor)
        new_height = int(image.height * upscale_factor)

        block_state.upscaled_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)
        block_state.upscaled_width = new_width
        block_state.upscaled_height = new_height

        logger.info(
            f"Upscaled image from {image.width}x{image.height} to {new_width}x{new_height} "
            f"(factor={upscale_factor})"
        )

        self.set_block_state(state, block_state)
        return components, state


class UltimateSDUpscaleTilePlanStep(ModularPipelineBlocks):
    """Plans the tile grid for the upscaled image.

    Generates a list of ``TileSpec`` objects based on the requested tile size
    and padding. Supports linear (raster) and chess (checkerboard) traversal.
    Optionally plans seam-fix bands along tile boundaries.
    """

    @property
    def description(self) -> str:
        return (
            "Tile planning step that generates tile coordinates for the upscaled image.\n"
            "Supports 'linear' (raster) and 'chess' (checkerboard) traversal.\n"
            "Optionally plans seam-fix bands along tile boundaries."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("upscaled_width", type_hint=int, required=True,
                       description="Width of the upscaled image."),
            InputParam("upscaled_height", type_hint=int, required=True,
                       description="Height of the upscaled image."),
            InputParam("tile_size", type_hint=int, default=2048,
                       description="Base tile size in pixels. Default 2048 processes most images "
                                   "in a single pass (seamless). Set to 512 for tiled mode on very large images."),
            InputParam("tile_padding", type_hint=int, default=32,
                       description="Number of overlap pixels on each side of a tile. Only relevant when tiling."),
            InputParam("traversal_mode", type_hint=str, default="linear",
                       description="Tile traversal order: 'linear' or 'chess'."),
            InputParam("seam_fix_width", type_hint=int, default=0,
                       description="Width of seam-fix bands in pixels. 0 disables seam fixing."),
            InputParam("seam_fix_padding", type_hint=int, default=16,
                       description="Extra padding around seam-fix bands for denoise context."),
            InputParam("seam_fix_mask_blur", type_hint=int, default=8,
                       description="Feathering width for seam-fix blending masks."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("tile_plan", type_hint=list,
                        description="List of TileSpec defining the tile grid."),
            OutputParam("num_tiles", type_hint=int,
                        description="Total number of tiles in the plan."),
            OutputParam("seam_fix_plan", type_hint=list,
                        description="List of SeamFixSpec for seam-fix bands (empty if disabled)."),
            OutputParam("seam_fix_mask_blur", type_hint=int,
                        description="Feathering width for seam-fix blending."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        tile_size = block_state.tile_size
        tile_padding = block_state.tile_padding
        traversal_mode = block_state.traversal_mode

        if traversal_mode not in ("linear", "chess"):
            raise ValueError(
                f"Unsupported traversal_mode '{traversal_mode}'. "
                "Supported modes: 'linear', 'chess'."
            )

        validate_tile_params(tile_size, tile_padding)

        if traversal_mode == "chess":
            tile_plan = plan_tiles_chess(
                image_width=block_state.upscaled_width,
                image_height=block_state.upscaled_height,
                tile_size=tile_size,
                tile_padding=tile_padding,
            )
        else:
            tile_plan = plan_tiles_linear(
                image_width=block_state.upscaled_width,
                image_height=block_state.upscaled_height,
                tile_size=tile_size,
                tile_padding=tile_padding,
            )

        # Validate and plan seam-fix bands if enabled
        seam_fix_width = block_state.seam_fix_width
        seam_fix_padding = block_state.seam_fix_padding
        seam_fix_mask_blur = block_state.seam_fix_mask_blur

        if seam_fix_width < 0:
            raise ValueError(f"`seam_fix_width` must be non-negative, got {seam_fix_width}.")
        if seam_fix_padding < 0:
            raise ValueError(f"`seam_fix_padding` must be non-negative, got {seam_fix_padding}.")
        if seam_fix_mask_blur < 0:
            raise ValueError(f"`seam_fix_mask_blur` must be non-negative, got {seam_fix_mask_blur}.")

        if seam_fix_width > 0:
            seam_fix_plan = plan_seam_fix_bands(
                tiles=tile_plan,
                image_width=block_state.upscaled_width,
                image_height=block_state.upscaled_height,
                seam_fix_width=seam_fix_width,
                seam_fix_padding=seam_fix_padding,
            )
        else:
            seam_fix_plan = []

        block_state.tile_plan = tile_plan
        block_state.num_tiles = len(tile_plan)
        block_state.seam_fix_plan = seam_fix_plan
        block_state.seam_fix_mask_blur = seam_fix_mask_blur

        logger.info(
            f"Planned {len(tile_plan)} tiles "
            f"(tile_size={tile_size}, padding={tile_padding}, traversal={traversal_mode})"
            + (f", {len(seam_fix_plan)} seam-fix bands" if seam_fix_plan else "")
        )

        self.set_block_state(state, block_state)
        return components, state
