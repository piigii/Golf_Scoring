"""LocateAnything golf-club boxes on EVERY frame, with resumable raw predictions."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
PROMPT = "Locate a single instance that matches the following description: golf club."


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_records(path, frame_count):
    records = {}
    if not path.exists():
        return records
    lines = path.read_bytes().splitlines(keepends=True)
    valid_bytes = 0
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            # A crash may leave only the final JSON line incomplete.
            with path.open("r+b") as out:
                out.truncate(valid_bytes)
            break
        f = record["frame"]
        if f in records or not 0 <= f < frame_count:
            raise ValueError("Duplicate or invalid cached frame")
        records[f] = record
        valid_bytes += len(line)
    return records


def put(image, value, xy, scale=.5, color=(235, 235, 235)):
    import cv2
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def decorate(frame, record, name, index, count, fps, backend_label="BF16 / MLX / hybrid"):
    import cv2
    import numpy as np
    panel = np.full((720, 960, 3), (28, 25, 22), np.uint8)
    panel[48:688, :640] = cv2.resize(frame, (640, 640))
    put(panel, f"VIDEO {index + 1}/{count} | {name} | frame {record['frame']} | {record['frame']/fps:.3f}s", (15, 30), .65)
    h, w = frame.shape[:2]
    color = (40, 220, 255)
    for b in record["boxes"]:
        x1, y1, x2, y2 = b["pixels_xyxy"]
        a = (round(x1 / w * 640), 48 + round(y1 / h * 640))
        z = (min(639, round(x2 / w * 640)), min(687, 48 + round(y2 / h * 640)))
        cv2.rectangle(panel, a, z, color, 2)
        put(panel, "golf club", (max(0, a[0]), max(65, a[1] - 7)), .48, color)
    put(panel, "LocateAnything-3B", (660, 79), .65)
    put(panel, backend_label, (660, 111), .48)
    put(panel, "VLM detection EVERY frame", (660, 152), .48)
    put(panel, f"Boxes: {len(record['boxes'])}", (660, 194), .6, color)
    if not record["boxes"]:
        put(panel, "NO VALID BOX", (660, 235), .55, (80, 130, 255))
    elif record["large_box_review"]:
        put(panel, "REVIEW: very large box", (660, 235), .48, (80, 130, 255))
    for k, box in enumerate(record["boxes"][:4]):
        x1, y1, x2, y2 = box["pixels_xyxy"]
        put(panel, f"Box {k + 1}: x1={x1:.1f} y1={y1:.1f}", (660, 278 + k * 53), .43)
        put(panel, f"       x2={x2:.1f} y2={y2:.1f}", (660, 299 + k * 53), .43)
    put(panel, "Raw model prediction", (660, 529), .5)
    put(panel, "No tracker/interpolation", (660, 562), .46)
    put(panel, "No calibrated confidence", (660, 595), .46)
    put(panel, "Thin/blurred clubs may be missed", (660, 628), .4)
    put(panel, "Yellow box = VLM golf-club prediction; source 160x160", (15, 709), .48)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT.parent / "videos_160")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/golf_club_first2")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--backend", choices=["auto", "metal", "cuda"], default="auto")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--image-token-limit", type=int, default=1024)
    parser.add_argument("--mode", choices=["hybrid", "fast", "slow"], default="hybrid")
    parser.add_argument("--codec", choices=["mp4v", "avc1"], default="mp4v")
    args = parser.parse_args()
    import cv2
    from PIL import Image
    from infer import parse_coordinates
    from backend import select_backend, default_model, model_identity, load_backend
    if min(args.count, args.max_tokens, args.image_token_limit) < 1:
        parser.error("Count and token limits must be positive")
    args.backend = select_backend(args.backend)
    args.model = args.model or default_model(args.backend)
    if not (args.model / "config.json").is_file():
        parser.error(f"Missing model. Run vlm/download_model.py --backend {args.backend}")
    files = sorted(args.input.glob("*.mp4"), key=lambda p: (int(p.stem) if p.stem.isdigit() else float("inf"), p.name))[:args.count]
    if len(files) != args.count:
        raise ValueError("Not enough input videos")
    args.output.mkdir(parents=True, exist_ok=True)
    print("Hashing model and source files for safe resume...", flush=True)
    settings = {**model_identity(args.model), "backend": args.backend,
                "prompt": PROMPT, "mode": args.mode,
                "max_tokens": args.max_tokens, "image_token_limit": args.image_token_limit,
                "codec": args.codec, "sampling": "every original frame",
                "sources": {p.name: sha256(p) for p in files}}
    print(f"Loading {args.backend} model: {args.model}", flush=True)
    engine = load_backend(args.backend, args.model, args.image_token_limit)
    settings.update(dtype=engine.dtype, hardware=engine.hardware)
    config_path = args.output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError("Cached run has different settings or source files; use a new output folder")
    config_path.write_text(json.dumps(settings, indent=2) + "\n")
    state = {"status": "running", "settings": settings, "clips": [], "current_clip": None}
    status_path = args.output / "batch_status.json"
    def status():
        status_path.write_text(json.dumps(state, indent=2) + "\n")
    status()
    label = f"{engine.dtype} / {engine.name} / {args.mode}"
    state.update(hardware=engine.hardware, backend=engine.name, dtype=engine.dtype)
    compiled_frames = 0
    try:
        for index, source in enumerate(files):
            cap = cv2.VideoCapture(str(source))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            if not frames or fps <= 0:
                raise ValueError(f"Invalid video: {source}")
            directory = args.output / source.stem
            directory.mkdir(exist_ok=True)
            jsonl = directory / "detections.jsonl"
            records = read_records(jsonl, len(frames))
            h, w = frames[0].shape[:2]
            print(f"[{index + 1}/{len(files)}] {source.name}: {len(frames)} frames; cached {len(records)}", flush=True)
            state["current_clip"] = source.name
            with jsonl.open("a") as out:
                for f, frame in enumerate(frames):
                    if f not in records:
                        started = time.perf_counter()
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        answer = engine.predict(image, PROMPT, args.max_tokens, args.mode)
                        detections = parse_coordinates(answer, w, h)
                        boxes = [b for b in detections if b["type"] == "box"
                                 and b["pixels_xyxy"][2] > b["pixels_xyxy"][0]
                                 and b["pixels_xyxy"][3] > b["pixels_xyxy"][1]]
                        review = any((b["pixels_xyxy"][2] - b["pixels_xyxy"][0]) *
                                     (b["pixels_xyxy"][3] - b["pixels_xyxy"][1]) > .8 * w * h for b in boxes)
                        record = {"frame": f, "time_seconds": f / fps, "raw_output": answer,
                                  "boxes": boxes, "large_box_review": review,
                                  "inference_seconds": time.perf_counter() - started}
                        out.write(json.dumps(record) + "\n")
                        out.flush()
                        records[f] = record
                    if f % 10 == 0 or f == len(frames) - 1:
                        state.update(current_frame=f, current_total=len(frames), current_predictions=len(records))
                        status()
                        cv2.imwrite(str(args.output / "latest_preview.jpg"), decorate(frame, records[f], source.name, index, len(files), fps, label))
                        print(f"{source.name}: {f + 1}/{len(frames)} | boxes={len(records[f]['boxes'])}", flush=True)
                        engine.clear_cache()
            assert len(records) == len(frames)
            writer = cv2.VideoWriter(str(directory / "annotated.mp4"), cv2.VideoWriter_fourcc(*args.codec), fps, (960, 720))
            if not writer.isOpened():
                raise RuntimeError("Cannot create output with selected codec")
            try:
                for f, frame in enumerate(frames):
                    rendered = decorate(frame, records[f], source.name, index, len(files), fps, label)
                    writer.write(rendered)
            finally:
                writer.release()
            summary = {"source": source.name, "fps": fps, "num_frames": len(frames),
                       "vlm_inferred_frames": len(records), "frames_with_box": sum(bool(r["boxes"]) for r in records.values()),
                       "large_box_review_frames": sum(r["large_box_review"] for r in records.values()),
                       "inference_seconds": sum(r["inference_seconds"] for r in records.values()),
                       "compilation_start_frame": compiled_frames, "compilation_start_seconds": compiled_frames / 30,
                       "compilation_frames": round(len(frames) / fps * 30)}
            (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            state["clips"].append(summary)
            compiled_frames += summary["compilation_frames"]
            status()
            print(f"Completed {source.name}", flush=True)
        destination = args.output / "golf_club_first2.mp4"
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*args.codec), 30, (960, 720))
        if not writer.isOpened():
            raise RuntimeError("Cannot create compilation")
        try:
            for clip in state["clips"]:
                cap = cv2.VideoCapture(str(args.output / Path(clip["source"]).stem / "annotated.mp4"))
                images = []
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    images.append(image)
                cap.release()
                assert len(images) == clip["num_frames"]
                for k in range(clip["compilation_frames"]):
                    f = min(len(images) - 1, round(k / 30 * clip["fps"]))
                    writer.write(images[f])
        finally:
            writer.release()
        cap = cv2.VideoCapture(str(destination))
        count = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            assert image.shape == (720, 960, 3)
            count += 1
        cap.release()
        assert count == compiled_frames
        state.update(status="completed", output=str(destination.resolve()), output_frames=count,
                     fps=30, duration_seconds=count / 30, **engine.memory())
        status()
        print(f"Verified {count} compiled frames: {destination}", flush=True)
    except BaseException as error:
        state.update(status="failed", error=str(error))
        status()
        raise


if __name__ == "__main__":
    main()
