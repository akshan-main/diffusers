"""Minimal example: Ultimate SD Upscale with Modular Diffusers (SDXL).

Usage:
    python examples/modular_diffusers/ultimate_sd_upscale.py \
        --image input.png \
        --prompt "high quality, highly detailed, 8k" \
        --output output.png

Requirements:
    pip install diffusers transformers accelerate torch
"""

import argparse

import torch
from PIL import Image

from diffusers.modular_pipelines.ultimate_sd_upscale import UltimateSDUpscaleBlocks


def main():
    parser = argparse.ArgumentParser(description="Ultimate SD Upscale (SDXL, Modular Diffusers)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="high quality, highly detailed, 8k")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, artifacts")
    parser.add_argument("--output", type=str, default="upscaled.png")
    parser.add_argument("--upscale_factor", type=float, default=2.0)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_padding", type=int, default=32)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    args = parser.parse_args()

    # Load input image
    image = Image.open(args.image).convert("RGB")
    print(f"Input: {image.width}x{image.height}")

    # Create pipeline from blocks
    blocks = UltimateSDUpscaleBlocks()
    pipe = blocks.init_pipeline(args.model)
    pipe.load_components(torch_dtype=torch.float16)
    pipe.to("cuda")

    # Run
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=image,
        upscale_factor=args.upscale_factor,
        tile_size=args.tile_size,
        tile_padding=args.tile_padding,
        strength=args.strength,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        output="images",
    )

    result[0].save(args.output)
    print(f"Output: {result[0].width}x{result[0].height} → {args.output}")


if __name__ == "__main__":
    main()
