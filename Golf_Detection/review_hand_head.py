"""Audit saved GolfPose head/wrist predictions; does not measure accuracy."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
TARGETS = [(10, "head", (0, 220, 255)), (13, "left_wrist", (255, 100, 40)),
           (16, "right_wrist", (220, 60, 240))]


def put(image, value, xy, color=(235, 235, 235), scale=.55):
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    destination = ROOT / "outputs/hand_head_review"
    destination.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination / "hand_head_tracking.mp4"),
                             cv2.VideoWriter_fourcc(*"avc1"), 30, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("Cannot create review video")
    rows, all_scores, all_jumps, all_outside = [], {}, {}, {}
    output_frames = 0
    try:
        for clip in range(20):
            directory = ROOT / f"outputs/first20/{clip}"
            data = np.load(directory / "pose2d.npz")
            report = json.loads((directory / "events.json").read_text())
            points, scores = data["keypoints"], data["scores"]
            width, height = report["width"], report["height"]
            for j, name, _ in TARGETS:
                assert str(data["joint_names"][j]) == name
                s = scores[:, j]
                jumps = np.linalg.norm(np.diff(points[:, j], axis=0), axis=-1)
                xy = points[:, j]
                outside = (xy[:, 0] < 0) | (xy[:, 0] >= width) | (xy[:, 1] < 0) | (xy[:, 1] >= height)
                all_scores.setdefault(name, []).extend(s.tolist())
                all_jumps.setdefault(name, []).extend(jumps.tolist())
                all_outside.setdefault(name, []).extend(outside.tolist())
                rows.append({"video": f"{clip}.mp4", "joint": name, "frames": len(s),
                             "median_heatmap_score": float(np.median(s)),
                             "below_0_3_percent": float(np.mean(s < .3) * 100),
                             "outside_image_percent": float(outside.mean() * 100),
                             "jump_p95_px_per_frame": float(np.percentile(jumps, 95)),
                             "jump_max_px_per_frame": float(jumps.max()),
                             "largest_jump_frame": int(jumps.argmax()) + 1})
            cap = cv2.VideoCapture(str(ROOT / f"videos_160/{clip}.mp4"))
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            assert len(frames) == len(points)
            for k in range(round(len(frames) / report["fps"] * 30)):
                f = min(len(frames) - 1, round(k / 30 * report["fps"]))
                frame = frames[f]
                panel = np.full((720, 1280, 3), (28, 25, 22), np.uint8)
                put(panel, f"{clip}.mp4 | frame {f} | {f / report['fps']:.3f}s | HEAD / WRIST AUDIT", (18, 29), scale=.7)
                panel[50:690, :640] = cv2.resize(frame, (640, 640))
                for row, (j, name, color) in enumerate(TARGETS):
                    x, y = points[f, j]
                    score = float(scores[f, j])
                    inside = 0 <= x < width and 0 <= y < height
                    if inside:
                        target = (round(float(x / width * 640)), round(float(y / height * 640)) + 50)
                        cv2.circle(panel, target, 7, color, 2)
                        put(panel, name, (max(0, min(490, target[0] + 9)), max(65, min(675, target[1] - 8))), color, .45)
                    # Raw low-score points remain visible for diagnosis.
                    y0 = 50 + row * 210
                    padded = cv2.copyMakeBorder(frame, 64, 64, 64, 64, cv2.BORDER_CONSTANT)
                    cx, cy = int(np.clip(round(float(x)), 0, width - 1)) + 64, int(np.clip(round(float(y)), 0, height - 1)) + 64
                    crop = padded[cy - 32:cy + 32, cx - 32:cx + 32]
                    crop = cv2.resize(crop, (200, 200))
                    if inside:
                        cv2.drawMarker(crop, (100, 100), color, cv2.MARKER_CROSS, 20, 1)
                    panel[y0:y0 + 200, 650:850] = crop
                    jump = float(np.linalg.norm(points[f, j] - points[f - 1, j])) if f else 0.
                    put(panel, name.upper(), (870, y0 + 25), color, .7)
                    put(panel, f"Heatmap score: {score:.3f}", (870, y0 + 60))
                    put(panel, f"XY: {x:.1f}, {y:.1f} px", (870, y0 + 91))
                    put(panel, f"Step: {jump:.1f} px/frame", (870, y0 + 122))
                    label = "OUTSIDE IMAGE" if not inside else ("LOW SCORE < 0.3" if score < .3 else "Score >= 0.3")
                    put(panel, label, (870, y0 + 158), (80, 120, 255) if not inside or score < .3 else color)
                put(panel, "Raw model predictions. Scores/jumps are diagnostics, not accuracy. No finger joints.", (18, 714), scale=.5)
                writer.write(panel)
                output_frames += 1
            print(f"Reviewed {clip}.mp4", flush=True)
    finally:
        writer.release()
    summary = {"source": "20 videos: 0.mp4 to 19.mp4", "frames_per_joint": len(all_scores["head"]),
               "threshold": .3, "score_type": "raw HRNet heatmap peak; not calibrated probability",
               "ground_truth_available": False, "joints": {}}
    for name in all_scores:
        s, d, outside = np.array(all_scores[name]), np.array(all_jumps[name]), np.array(all_outside[name])
        summary["joints"][name] = {"median_heatmap_score": float(np.median(s)),
            "below_0_3_percent": float(np.mean(s < .3) * 100),
            "outside_image_percent": float(outside.mean() * 100),
            "jump_p95_px_per_frame": float(np.percentile(d, 95)),
            "jump_max_px_per_frame": float(d.max())}
    cap = cv2.VideoCapture(str(destination / "hand_head_tracking.mp4"))
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (720, 1280, 3)
        decoded += 1
    cap.release()
    assert decoded == output_frames == summary["frames_per_joint"]
    summary["review_video"] = {"frames": decoded, "fps": 30, "size": [1280, 720]}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (destination / "per_video.csv").open("w", newline="") as out:
        writer_csv = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
