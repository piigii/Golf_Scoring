"""Run real GolfPose/SwingNet inference and compile the first numbered videos."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numbered_key(path):
    return (int(path.stem) if path.stem.isdigit() else float("inf"), path.name)


def validate_clip(directory, source):
    report = json.loads((directory / "events.json").read_text())
    if report["video_sha256"] != sha256(source):
        raise ValueError(f"Source changed: {source}")
    a, b = np.load(directory / "pose2d.npz"), np.load(directory / "pose3d.npz")
    n = report["num_frames"]
    assert a["keypoints"].shape == (n, 22, 2)
    assert b["keypoints"].shape == (n, 22, 3)
    assert np.isfinite(a["keypoints"]).all() and np.isfinite(b["keypoints"]).all()
    assert np.array_equal(a["timestamps"], b["timestamps"])
    for event in report["events"]:
        f = event["frame"]
        np.testing.assert_allclose(event["pose_2d"], a["keypoints"][f])
        np.testing.assert_allclose(event["pose_3d_root_relative_m"], b["keypoints"][f])
    cap = cv2.VideoCapture(str(directory / "annotated.mp4"))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (480, 960, 3)
        frames.append(frame)
    cap.release()
    assert len(frames) == n, (source, len(frames), n)
    return report, frames


def text(image, value, xy, scale=.5, color=(225, 225, 225)):
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def decorate(frame, report, source_name, clip_index, count, original_frame):
    panel = np.full((600, 1280, 3), (28, 25, 22), dtype=np.uint8)
    panel[48:528, :960] = frame
    time = original_frame / report["fps"]
    text(panel, f"VIDEO {clip_index + 1:02d}/{count:02d} | {source_name} | {time:.2f}s", (20, 31), .7)
    text(panel, "2D: body + club keypoints", (20, 551), .5)
    text(panel, "3D: camera-relative estimate", (490, 551), .5)
    text(panel, "8 SWING EVENTS - SwingNet", (975, 32), .52)
    text(panel, "Frame / time / confidence", (975, 59), .43)
    for event in report["events"]:
        y = 78 + event["label"] * 55
        # Hold the highlight around the predicted instant for readability.
        active = abs(original_frame - event["frame"]) <= max(1, round(.18 * report["fps"]))
        color = (65, 220, 255) if active else (220, 220, 220)
        if active:
            cv2.rectangle(panel, (966, y - 10), (1270, y + 39), (70, 65, 42), -1)
        text(panel, f"{event['label'] + 1}. {event['name']}", (976, y + 6), .48, color)
        text(panel, f"f{event['frame']} | {event['time_seconds']:.2f}s | {event['confidence']:.3f}",
             (976, y + 27), .43, color)
    text(panel, "Green: body  Orange: club", (20, 580), .45)
    text(panel, "3D/club and low-confidence events need review", (490, 580), .42)
    if not report["events_strictly_ordered"]:
        text(panel, "Check event order (raw predictions)", (975, 548), .42, (65, 170, 255))
    duration = max((report["num_frames"] - 1) / report["fps"], 1e-6)
    x0, x1, y = 978, 1258, 577
    cv2.line(panel, (x0, y), (x1, y), (130, 130, 130), 2)
    for event in report["events"]:
        x = x0 + round((x1 - x0) * event["time_seconds"] / duration)
        cv2.circle(panel, (min(x1, max(x0, x)), y), 4, (65, 220, 255), -1)
    x = x0 + round((x1 - x0) * time / duration)
    cv2.line(panel, (min(x1, x), y - 8), (min(x1, x), y + 8), (255, 255, 255), 2)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "videos_160")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/first20")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--codec", choices=["mp4v", "avc1"], default="mp4v")
    args = parser.parse_args()
    files = sorted(args.input.glob("*.mp4"), key=numbered_key)[:args.count]
    if len(files) != args.count or args.count <= 0:
        parser.error("Not enough input videos, or invalid count")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "running", "sort": "numeric filename", "fps": 30,
               "size": [1280, 600], "annotation": "model predictions, not manual ground truth",
               "clips": []}
    status_path = args.output / "batch_status.json"
    def save_status():
        status_path.write_text(json.dumps(summary, indent=2) + "\n")
    save_status()
    destination = args.output / "golf_first20.mp4"
    # mp4v works with the OpenCV wheels on Linux and macOS.
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*args.codec), 30, (1280, 600))
    if not writer.isOpened():
        raise RuntimeError("Cannot create compilation with selected codec")
    frame_count = 0
    try:
        for index, source in enumerate(files):
            directory = args.output / source.stem
            directory.mkdir(exist_ok=True)
            print(f"[{index + 1}/{len(files)}] {source.name}", flush=True)
            if not (directory / "annotated.mp4").is_file():
                with (directory / "run.log").open("w") as log:
                    subprocess.run([sys.executable, str(ROOT / "golf_pipeline.py"),
                                    "--video", str(source.resolve()), "--output", str(directory.resolve()),
                                    "--device", args.device], stdout=log, stderr=subprocess.STDOUT, check=True)
            report, frames = validate_clip(directory, source)
            # Preserve duration at a common output FPS. Original pose/event indices stay unchanged.
            length = round(len(frames) / report["fps"] * 30)
            for k in range(length):
                original_frame = min(len(frames) - 1, round(k / 30 * report["fps"]))
                writer.write(decorate(frames[original_frame], report, source.name,
                                      index, len(files), original_frame))
            summary["clips"].append({"source": str(source.resolve()), "source_frames": len(frames),
                "source_fps": report["fps"], "output_start_frame": frame_count,
                "output_start_seconds": frame_count / 30, "output_frames": length,
                "events_strictly_ordered": report["events_strictly_ordered"],
                "events": [{k: e[k] for k in ("label", "name", "frame", "time_seconds", "confidence")}
                           for e in report["events"]]})
            frame_count += length
            save_status()
            print(f"Completed {source.name}: {len(frames)} frames; compiled {frame_count} frames", flush=True)
    except BaseException as error:
        summary.update(status="failed", error=str(error))
        save_status()
        raise
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(destination))
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (600, 1280, 3)
        decoded += 1
    cap.release()
    assert decoded == frame_count, (decoded, frame_count)
    summary.update(status="completed", output=str(destination.resolve()), output_frames=decoded,
                   duration_seconds=decoded / 30, output_sha256=sha256(destination))
    save_status()
    print(f"Verified {decoded} output frames: {destination}", flush=True)


if __name__ == "__main__":
    main()
