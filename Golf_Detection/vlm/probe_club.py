"""Compare native PBD hybrid/fast decoding on representative golf frames."""
import json
import os
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='Locate a single instance that matches the following description: the golf club held by the golfer, including its shaft and club head.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/golf_club_first20/probe')
    parser.add_argument('--modes', nargs='+', default=['hybrid', 'fast'])
    args = parser.parse_args()
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    from PIL import Image, ImageDraw
    from checkpoint_compat import install_checkpoint_aliases
    from infer import parse_coordinates

    install_checkpoint_aliases()
    model, processor = load(str(ROOT / "models/LocateAnything-3B-MLX"), strict=True)
    processor.image_processor.in_token_limit = 1024
    prompt_text = args.prompt
    prompt = apply_chat_template(processor, model.config, prompt_text, num_images=1)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((ROOT / "examples/club_probe").glob("*.jpg")):
        for mode in args.modes:
            started = time.perf_counter()
            inputs = prepare_inputs(processor, images=[str(path)], prompts=prompt)
            input_ids = inputs.pop("input_ids")
            inputs.pop("attention_mask", None)
            tokens = model.pbd_generate(input_ids, generation_mode=mode, max_tokens=48, **inputs)
            answer = processor.decode(tokens, skip_special_tokens=False)
            elapsed = time.perf_counter() - started
            image = Image.open(path).convert("RGB")
            detections = parse_coordinates(answer, *image.size)
            record = {"image": path.name, "mode": mode, "seconds": elapsed,
                      "raw_output": answer, "detections": detections}
            records.append(record)
            print(json.dumps(record), flush=True)
            draw = ImageDraw.Draw(image)
            for d in detections:
                if d["type"] == "box":
                    draw.rectangle(d["pixels_xyxy"], outline="red", width=1)
            image.resize((640, 640)).save(output / f"{path.stem}_{mode}.jpg")
            (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            mx.clear_cache()


if __name__ == "__main__":
    main()
