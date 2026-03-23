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

"""Modular pipeline class for Ultimate SD Upscale.

Reuses ``StableDiffusionXLModularPipeline`` since all components (VAE, UNet,
text encoders, scheduler) are the same.  The only addition is the
``default_blocks_name`` pointing to our custom block composition.
"""

from ..stable_diffusion_xl.modular_pipeline import StableDiffusionXLModularPipeline


class UltimateSDUpscaleModularPipeline(StableDiffusionXLModularPipeline):
    """A ModularPipeline for Ultimate SD Upscale (SDXL).

    Inherits all SDXL component properties (``vae_scale_factor``,
    ``default_sample_size``, etc.) and overrides the default blocks to use
    the Ultimate SD Upscale block composition.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "UltimateSDUpscaleBlocks"
