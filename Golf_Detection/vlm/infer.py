"""Single-image LocateAnything inference on Apple Metal or NVIDIA CUDA."""
import argparse
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_coordinates(answer, width, height):
    """NVIDIA output order is x/y, normalized to 0..1000; preserve raw output too."""
    detections = []
    for block in re.findall(r"<box>(.*?)</box>", answer, flags=re.DOTALL):
        values = [int(v) for v in re.findall(r"<(\d+)>", block)]
        if any(v > 1000 for v in values):
            continue
        if len(values) == 4:
            x1, y1, x2, y2 = values
            if x2 < x1 or y2 < y1:
                continue
            detections.append({"type": "box", "normalized_xyxy": values,
                               "pixels_xyxy": [x1 * width / 1000, y1 * height / 1000,
                                               x2 * width / 1000, y2 * height / 1000]})
        elif len(values) == 2:
            x, y = values
            detections.append({"type": "point", "normalized_xy": values,
                               "pixels_xy": [x * width / 1000, y * height / 1000]})
    return detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="Locate a single instance that matches the following description: the golfer.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/latest")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--image-token-limit", type=int, default=1024)
    parser.add_argument("--backend", choices=["auto", "metal", "cuda"], default="auto")
    parser.add_argument("--mode", choices=["hybrid", "fast", "slow"], default="hybrid")
    args = parser.parse_args()
    from backend import select_backend, default_model, load_backend
    args.backend = select_backend(args.backend)
    args.model = args.model or default_model(args.backend)
    if not args.image.is_file() or not (args.model / "config.json").is_file():
        parser.error("Image or downloaded model missing. Run vlm/download_model.py first.")
    if args.max_tokens <= 0 or args.image_token_limit <= 0:
        parser.error("Token limits must be positive")
    from PIL import Image, ImageDraw

    args.output.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    started = time.perf_counter()
    print(f"Loading {args.model.resolve()}", flush=True)
    engine = load_backend(args.backend, args.model, args.image_token_limit)
    loaded = time.perf_counter()
    print("Running single-image inference...", flush=True)
    answer = engine.predict(image, args.prompt, args.max_tokens, args.mode)
    ended = time.perf_counter()
    detections = parse_coordinates(answer, *image.size)
    report = {"image": str(args.image.resolve()), "size": list(image.size),
              "model": str(args.model.resolve()), "prompt": args.prompt,
              "image_token_limit": args.image_token_limit, "max_tokens": args.max_tokens,
              "raw_output": answer, "detections": detections,
              "load_seconds": loaded - started, "inference_seconds": ended - loaded,
              "backend": args.backend, "dtype": engine.dtype, "hardware": engine.hardware,
              "mode": args.mode, **engine.memory()}
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "response.txt").write_text(answer + "\n")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for detection in detections:
        if detection["type"] == "box":
            draw.rectangle(detection["pixels_xyxy"], outline="red", width=3)
        else:
            x, y = detection["pixels_xy"]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline="red", width=2)
    overlay.save(args.output / "annotated.jpg")
    print(json.dumps(report, indent=2), flush=True)
    if not detections:
        print("No valid box/point parsed. See raw_output; inference does not guarantee a detection.", flush=True)


if __name__ == "__main__":
    main()
