"""GolfPose 2D -> GolfPose MixSTE 3D -> SwingNet events on one cropped swing.

Uses actual pretrained checkpoints, never untrained or substitute pose models.
All stages retain the original decoded frame indices (zero based).
"""
import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from model import EventDetector
from device import select_device

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
EVENT_NAMES = ["Address", "Toe-up", "Mid-backswing", "Top", "Mid-downswing",
               "Impact", "Mid-follow-through", "Finish"]
BODY_EDGES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7),
              (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
              (8, 14), (14, 15), (15, 16)]
CLUB_EDGES = [(17, 18), (18, 19), (19, 20), (19, 21)]


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, timestamps = [], []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    finally:
        cap.release()
    if not frames or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video is empty or has invalid FPS.")
    if len({f.shape for f in frames}) != 1:
        raise ValueError("Video frame dimensions must remain constant.")
    # Some codecs do not provide timestamps. CFR fallback is explicitly recorded.
    timestamps = np.asarray(timestamps, dtype=np.float64)
    measured = len(timestamps) > 1 and np.all(np.diff(timestamps) > 0)
    if not measured:
        timestamps = np.arange(len(frames), dtype=np.float64) / fps
    return frames, float(fps), timestamps, "decoder" if measured else "frame_index/fps"


def pose_2d(frames, repo, config, checkpoint, device):
    # Import only when this stage is selected, so saved 2D data can be reused.
    from mmengine.config import Config
    from golfpose_hrnet import infer_hrnet

    cfg = Config.fromfile(str(config))
    meta = repo / cfg.metainfo["from_file"]
    info = runpy.run_path(str(meta))["dataset_info"]
    # test_video is already a single cropped golfer; use the full frame bbox.
    return infer_hrnet(frames, cfg, info, checkpoint, device)


def pose_3d(points, width, height, repo, checkpoint, device):
    sys.path.insert(0, str(repo))
    from common.model_cross import MixSTE2
    from common.camera import normalize_screen_coordinates

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {k.removeprefix("module."): v for k, v in saved["model_pos"].items()}
    num_joints = state["Spatial_pos_embed"].shape[1]
    window = state["Temporal_pos_embed"].shape[1]
    channels = state["Spatial_pos_embed"].shape[2]
    depth = max(int(k.split(".")[1]) for k in state if k.startswith("STEblocks.")) + 1
    if points.shape[1] < num_joints:
        raise ValueError(f"3D model needs {num_joints} joints; 2D model provided {points.shape[1]}.")
    model = MixSTE2(num_frame=window, num_joints=num_joints, in_chans=2,
                    embed_dim_ratio=channels, depth=depth, num_heads=8,
                    mlp_ratio=2., qkv_bias=True, drop_path_rate=0.)
    # Strict loading prevents silently rendering an untrained/mismatched model.
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    normalized = normalize_screen_coordinates(points[:, :num_joints], width, height)
    normalized = normalized.astype(np.float32)
    total = len(normalized)
    summed = np.zeros((total, num_joints, 3), dtype=np.float32)
    counts = np.zeros((total, 1, 1), dtype=np.float32)
    stride = max(1, window // 2)
    starts = list(range(0, max(1, total - window + 1), stride))
    final_start = max(0, total - window)
    if starts[-1] != final_start:
        starts.append(final_start)
    left, right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
    with torch.inference_mode():
        for start in starts:
            valid = min(window, total - start)
            clip = normalized[start:start + valid]
            clip = np.pad(clip, ((0, window - valid), (0, 0), (0, 0)), mode="edge")
            flipped = clip.copy()
            flipped[..., 0] *= -1
            flipped[:, left + right] = flipped[:, right + left]
            inputs = torch.from_numpy(np.stack([clip, flipped])).to(device)
            pred = model(inputs).cpu().numpy()
            pred[1, ..., 0] *= -1
            pred[1, :, left + right] = pred[1, :, right + left].copy()
            averaged = pred.mean(axis=0)[:valid]
            averaged -= averaged[:, :1].copy()
            summed[start:start + valid] += averaged
            counts[start:start + valid] += 1
            print(f"GolfPose 3D: frames {start}..{start + valid - 1}", flush=True)
    return summed / counts


def swingnet(frames, checkpoint, device, seq_length=64):
    model = EventDetector(pretrain=False, width_mult=1., lstm_layers=1,
                          lstm_hidden=256, bidirectional=True, dropout=False)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device).eval()
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(frames), seq_length):
            images = []
            for frame in frames[start:start + seq_length]:
                h, w = frame.shape[:2]
                scale = 160 / max(h, w)
                nh, nw = int(h * scale), int(w * scale)
                image = cv2.resize(frame, (nw, nh))
                dh, dw = 160 - nh, 160 - nw
                image = cv2.copyMakeBorder(image, dh // 2, dh - dh // 2,
                                          dw // 2, dw - dw // 2,
                                          cv2.BORDER_CONSTANT, value=(103.53, 116.28, 123.675))
                image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - mean) / std
                images.append(image.transpose(2, 0, 1))
            tensor = torch.from_numpy(np.stack(images)).unsqueeze(0).to(device)
            probabilities.append(model(tensor).softmax(dim=-1).cpu().numpy())
    return np.concatenate(probabilities)


def event_records(probs, timestamps):
    if probs.shape != (len(timestamps), 9) or not np.isfinite(probs).all():
        raise ValueError("Expected finite SwingNet probabilities with shape (frames, 9).")
    indices = probs[:, :8].argmax(axis=0)
    events = []
    for label, frame in enumerate(indices):
        frame = int(frame)
        events.append({"label": label, "name": EVENT_NAMES[label], "frame": frame,
                       "time_seconds": float(timestamps[frame]),
                       "confidence": float(probs[frame, label])})
    return events, bool(np.all(np.diff(indices) > 0))


def make_events(probs, timestamps, poses_2d, scores, poses_3d):
    if not (len(probs) == len(poses_2d) == len(scores) == len(poses_3d)):
        raise ValueError("All stages must contain the same original video frames.")
    events, ordered = event_records(probs, timestamps)
    for event in events:
        frame = event["frame"]
        event.update(pose_2d=poses_2d[frame].tolist(),
                     pose_2d_scores=scores[frame].tolist(),
                     pose_3d_root_relative_m=poses_3d[frame].tolist())
    return events, ordered


def render_events_only(frames, events, output):
    tiles = []
    for event in events:
        image = frames[event["frame"]].copy()
        h, w = image.shape[:2]
        scale = min(320 / w, 280 / h)
        image = cv2.resize(image, (round(w * scale), round(h * scale)))
        tile = np.full((360, 320, 3), 245, np.uint8)
        ih, iw = image.shape[:2]
        tile[80:80 + ih, (320 - iw) // 2:(320 - iw) // 2 + iw] = image
        for line, text in enumerate([event["name"], f"Frame {event['frame']} | {event['time_seconds']:.3f}s",
                                     f"Confidence {event['confidence']:.3f}"]):
            cv2.putText(tile, text, (10, 20 + 23 * line), cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 30, 30), 1)
        tiles.append(tile)
        cv2.imwrite(str(output / f"event_{event['label']}.jpg"), tile)
    montage = np.concatenate([np.concatenate(tiles[:4], axis=1), np.concatenate(tiles[4:], axis=1)])
    cv2.imwrite(str(output / "events_montage.jpg"), montage)


def render(frames, points, scores, poses, events, fps, destination, score_threshold):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(4.8, 4.8), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    # Display camera x/y/z as horizontal/depth/vertical, without claiming world calibration.
    display = poses[..., [0, 2, 1]].copy()
    display[..., 2] *= -1
    radius = max(float(np.quantile(np.abs(display), .995)), .5) * 1.1
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius),
           xlabel="Camera X (m)", ylabel="Camera Z (m)", zlabel="-Camera Y (m)")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=-65)
    edges = BODY_EDGES + (CLUB_EDGES if poses.shape[1] == 22 else [])
    lines = [ax.plot([], [], [], color="#168bcb" if b < 17 else "#e58c18", lw=2)[0]
             for a, b in edges]
    scatter = ax.scatter([], [], [], s=8, c="#333333")
    writer = cv2.VideoWriter(str(destination / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (960, 480))
    if not writer.isOpened():
        raise RuntimeError("Cannot create annotated.mp4")
    event_lookup = {}
    for event in events:
        event_lookup.setdefault(event["frame"], []).append(event)
    snapshots = {}
    try:
        for i, frame in enumerate(frames):
            overlay = frame.copy()
            edges_2d = BODY_EDGES + (CLUB_EDGES if points.shape[1] == 22 else [])
            for a, b in edges_2d:
                if min(scores[i, a], scores[i, b]) >= score_threshold:
                    color = (30, 180, 255) if b >= 17 else (40, 220, 40)
                    cv2.line(overlay, tuple(np.rint(points[i, a]).astype(int)),
                             tuple(np.rint(points[i, b]).astype(int)), color, 2)
            for joint, score in zip(points[i], scores[i]):
                if score >= score_threshold:
                    cv2.circle(overlay, tuple(np.rint(joint).astype(int)), 3, (40, 50, 255), -1)
            h, w = overlay.shape[:2]
            scale = min(480 / w, 440 / h)
            resized = cv2.resize(overlay, (round(w * scale), round(h * scale)))
            panel = np.full((480, 480, 3), 245, dtype=np.uint8)
            rh, rw = resized.shape[:2]
            panel[40:40 + rh, (480 - rw) // 2:(480 - rw) // 2 + rw] = resized
            title = f"Frame {i} | {i / fps:.3f}s"
            if i in event_lookup:
                title += " | " + ", ".join(e["name"] for e in event_lookup[i])
            cv2.putText(panel, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .43, (30, 30, 30), 1)
            for line, (a, b) in zip(lines, edges):
                pair = display[i, [a, b]]
                line.set_data_3d(pair[:, 0], pair[:, 1], pair[:, 2])
            scatter._offsets3d = tuple(display[i, :, k] for k in range(3))
            ax.set_title("GolfPose 3D: root-relative estimate")
            canvas.draw()
            plot = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            combined = np.concatenate([panel, plot], axis=1)
            writer.write(combined)
            if i in event_lookup:
                for event in event_lookup[i]:
                    snapshots[event["label"]] = combined.copy()
                    cv2.imwrite(str(destination / f"event_{event['label']}_{event['name'].lower()}.jpg"), combined)
    finally:
        writer.release()
        fig.clear()
    montage = np.concatenate([cv2.resize(snapshots[i], (640, 320)) for i in range(8)], axis=0)
    cv2.imwrite(str(destination / "events_montage.jpg"), montage)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "test_video.mp4")
    parser.add_argument("--golfpose-dir", type=Path, default=ROOT / "third_party/GolfPose")
    parser.add_argument("--pose-config", type=Path, default=None)
    parser.add_argument("--pose2d-checkpoint", type=Path, default=ROOT / "models/golfpose_golfer_hrnetw48.pth")
    parser.add_argument("--pose3d-checkpoint", type=Path, default=ROOT / "models/golfpose_17+5_39.2_32.3_62.8.bin")
    parser.add_argument("--swingnet-checkpoint", type=Path, default=ROOT / "models/swingnet_1800.pth.tar")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/test_video")
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--score-threshold", type=float, default=.3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--swingnet-only", action="store_true",
                        help="Run only event detection; does not produce pose results.")
    args = parser.parse_args()
    if args.seq_length <= 0 or args.threads <= 0:
        parser.error("seq-length and threads must be positive")
    repo = args.golfpose_dir.resolve()
    config = args.pose_config or repo / "configs/mmpose/golfpose_golfer_hrnetw48.py"
    assets = ([args.video, args.swingnet_checkpoint] if args.swingnet_only else
              [args.video, config, args.pose2d_checkpoint, args.pose3d_checkpoint, args.swingnet_checkpoint])
    missing = [str(p) for p in assets if not p.is_file()]
    if missing:
        parser.error("Missing assets:\n" + "\n".join(missing) + "\nSee PIPELINE.md.")
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    frames, fps, timestamps, timing = read_video(args.video)
    print(f"Video: {len(frames)} frames, {fps:.3f} FPS; device: {device}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.swingnet_only:
        probs = swingnet(frames, args.swingnet_checkpoint, device, args.seq_length)
        events, ordered = event_records(probs, timestamps)
        np.save(args.output / "swingnet_probabilities.npy", probs)
        metadata = {"stage": "swingnet_only", "pose2d_inference_run": False,
                    "pose3d_inference_run": False, "video": str(args.video.resolve()),
                    "video_sha256": sha256(args.video), "fps": fps, "num_frames": len(frames),
                    "timestamp_source": timing, "frame_index_base": 0,
                    "events_strictly_ordered": ordered, "events": events,
                    "checkpoint_sha256": sha256(args.swingnet_checkpoint)}
        (args.output / "events.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        render_events_only(frames, events, args.output)
        for event in events:
            print(f"{event['name']:20} frame={event['frame']:4} time={event['time_seconds']:.3f}s confidence={event['confidence']:.3f}")
        print(f"SwingNet-only results saved to {args.output.resolve()}")
        return
    points, scores, names = pose_2d(frames, repo, config, args.pose2d_checkpoint, device)
    np.savez_compressed(args.output / "pose2d.npz", keypoints=points, scores=scores,
                        timestamps=timestamps, joint_names=np.asarray(names))
    h, w = frames[0].shape[:2]
    poses = pose_3d(points, w, h, repo, args.pose3d_checkpoint, device)
    np.savez_compressed(args.output / "pose3d.npz", keypoints=poses, timestamps=timestamps,
                        joint_names=np.asarray(names[:poses.shape[1]]), units="meters",
                        coordinate_system="root-relative camera coordinates")
    probs = swingnet(frames, args.swingnet_checkpoint, device, args.seq_length)
    np.save(args.output / "swingnet_probabilities.npy", probs)
    events, ordered = make_events(probs, timestamps, points, scores, poses)
    metadata = {"video": str(args.video.resolve()), "video_sha256": sha256(args.video),
                "fps": fps, "num_frames": len(frames), "width": w, "height": h,
                "timestamp_source": timing, "frame_index_base": 0, "device": str(device),
                "pose_2d_joint_names": names, "pose_3d_joint_names": names[:poses.shape[1]],
                "pose_3d_units": "meters", "pose_3d_coordinates": "root-relative camera coordinates",
                "event_decoder": "independent argmax, original SwingNet behavior",
                "events_strictly_ordered": ordered,
                "checkpoints": {p.name: sha256(p) for p in assets[2:]}, "events": events}
    (args.output / "events.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    render(frames, points, scores, poses, events, fps, args.output, args.score_threshold)
    for event in events:
        print(f"{event['name']:20} frame={event['frame']:4} time={event['time_seconds']:.3f}s confidence={event['confidence']:.3f}")
    if not ordered:
        print("WARNING: Independent SwingNet predictions are not strictly ordered; inspect events.json.")
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
