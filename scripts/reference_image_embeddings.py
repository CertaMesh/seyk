#!/usr/bin/env python3
"""Generate EmbeddingGemma 2 image inputs and fp32 reference embeddings.

This script intentionally keeps the complete padded pixel tensor and position
tensor in JSON so the Rust preprocessing path can be compared independently of
the Python model. Run it with the scratchpad virtualenv and HF_HUB_OFFLINE=1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer
from transformers.models.gemma4.image_processing_gemma4 import (
    get_aspect_ratio_preserving_size,
)


MODEL_DIR = Path(
    "/Users/krishankoenig/.cache/huggingface/hub/models--google--embeddinggemma-2"
    "/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b"
)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_images(out_dir: Path) -> list[tuple[str, Image.Image]]:
    """Create distinct deterministic images with deliberately varied ratios."""

    out_dir.mkdir(parents=True, exist_ok=True)

    images: list[tuple[str, Image.Image]] = []

    # Square icon: a red circle on a pale blue background.
    image = Image.new("RGB", (192, 192), (225, 240, 250))
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 40, 152, 152), fill=(220, 35, 35), outline=(90, 10, 10), width=4)
    draw.text((56, 12), "CIRCLE", fill=(15, 45, 70), font=_font(18))
    images.append(("red_circle_square", image))

    # Wide chart: bars with a simple trend and labels.
    image = Image.new("RGB", (480, 192), "white")
    draw = ImageDraw.Draw(image)
    draw.line((48, 152, 444, 152), fill=(35, 35, 35), width=3)
    draw.line((48, 32, 48, 152), fill=(35, 35, 35), width=3)
    values = [48, 82, 64, 118, 96, 143]
    colors = [(45, 110, 205), (55, 145, 210), (70, 175, 145), (245, 165, 45), (230, 100, 65), (170, 75, 170)]
    for i, (value, color) in enumerate(zip(values, colors)):
        x = 68 + i * 60
        draw.rectangle((x, 152 - value, x + 34, 152), fill=color)
        draw.text((x + 2, 158), str(i + 1), fill=(35, 35, 35), font=_font(14))
    draw.text((174, 10), "BAR CHART", fill=(25, 25, 25), font=_font(22))
    images.append(("bar_chart_wide", image))

    # Tall document-like page with text and ruled lines.
    image = Image.new("RGB", (192, 480), (248, 246, 238))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, 174, 462), outline=(80, 80, 75), width=3)
    draw.text((38, 38), "NOTES", fill=(30, 55, 80), font=_font(22))
    draw.line((38, 74, 154, 74), fill=(100, 120, 140), width=2)
    for row in range(9):
        y = 102 + row * 34
        draw.text((34, y - 4), f"{row + 1:02d}", fill=(150, 80, 50), font=_font(13))
        draw.line((62, y + 7, 156, y + 7), fill=(105, 105, 105), width=2)
    images.append(("text_page_tall", image))

    # A locally generated landscape-like scene, without any downloaded photo.
    width, height = 320, 224
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            if y < 138:
                pixels[x, y] = (80 + y // 4, 165 + y // 8, 225)
            else:
                pixels[x, y] = (30, 105 + (y - 138) // 3, 50)
    draw = ImageDraw.Draw(image)
    draw.polygon([(0, 154), (62, 82), (132, 154)], fill=(45, 70, 92))
    draw.polygon([(94, 154), (190, 60), (315, 154)], fill=(55, 80, 102))
    draw.polygon([(142, 154), (190, 60), (234, 154)], fill=(210, 220, 225))
    draw.ellipse((244, 28, 278, 62), fill=(255, 235, 120))
    draw.text((18, 184), "MOUNTAIN LANDSCAPE", fill=(245, 245, 225), font=_font(16))
    images.append(("mountain_landscape", image))

    for name, image in images:
        image.save(out_dir / f"{name}.png", format="PNG")
    return images


def dump_image(model: SentenceTransformer, name: str, image_path: Path) -> dict:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    processor = model.processor
    image_processor = processor.image_processor
    max_patches = image_processor.max_soft_tokens * image_processor.pooling_kernel_size**2
    resized_height, resized_width = get_aspect_ratio_preserving_size(
        height=height,
        width=width,
        patch_size=image_processor.patch_size,
        max_patches=max_patches,
        pooling_kernel_size=image_processor.pooling_kernel_size,
    )
    encoded = processor(images=image, return_tensors="pt")
    embedding = model.encode(
        {"image": str(image_path)},
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    input_ids = encoded["input_ids"][0].tolist()
    pixel_values = encoded["pixel_values"][0].cpu().numpy().astype(np.float32)
    position_ids = encoded["image_position_ids"][0].cpu().numpy().astype(np.int64)
    soft_tokens = int(np.count_nonzero(np.all(position_ids >= 0, axis=1)) // (image_processor.pooling_kernel_size**2))
    return {
        "name": name,
        "path": str(image_path),
        "original_size": [width, height],
        "resized_size": [resized_width, resized_height],
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values": pixel_values.tolist(),
        "position_ids_shape": list(position_ids.shape),
        "position_ids": position_ids.tolist(),
        "num_soft_tokens": soft_tokens,
        "input_ids": input_ids,
        "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="JSON file to write")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_dir = args.output.parent / "image_fixtures"
    images = make_images(image_dir)

    model = SentenceTransformer(
        str(MODEL_DIR),
        model_kwargs={"dtype": torch.float32},
        local_files_only=True,
    )
    records = [dump_image(model, name, image_dir / f"{name}.png") for name, _ in images]
    result = {
        "model": "google/embeddinggemma-2",
        "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
        "dtype": "float32",
        "patch_size": model.processor.image_processor.patch_size,
        "pooling_kernel_size": model.processor.image_processor.pooling_kernel_size,
        "max_soft_tokens": model.processor.image_processor.max_soft_tokens,
        "images": records,
    }
    args.output.write_text(json.dumps(result, separators=(",", ":")))
    print(f"wrote {args.output} ({len(records)} images)")


if __name__ == "__main__":
    main()
