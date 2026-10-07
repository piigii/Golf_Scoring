"""Export saved VLM timings on Metal or CUDA without rerunning inference."""
import argparse
import base64
import csv
import html
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))


def summarize(name, records, duration):
    import numpy as np
    values = np.array([r["inference_seconds"] for r in records], dtype=float)
    if not len(values) or not np.all(np.isfinite(values)) or not np.all(values > 0) or duration <= 0:
        raise ValueError("Frame timings and source duration must be finite and positive")
    return dict(video=name, frames=len(records), source_duration_seconds=duration,
                inference_seconds=float(values.sum()), mean_seconds_per_frame=float(values.mean()),
                median_seconds_per_frame=float(np.median(values)),
                p90_seconds_per_frame=float(np.percentile(values, 90)),
                p95_seconds_per_frame=float(np.percentile(values, 95)),
                p99_seconds_per_frame=float(np.percentile(values, 99)),
                min_seconds_per_frame=float(values.min()), max_seconds_per_frame=float(values.max()),
                throughput_frames_per_second=len(values)/float(values.sum()),
                slower_than_realtime=float(values.sum())/duration,
                frames_with_valid_box=sum(bool(r["boxes"]) for r in records),
                large_box_review_frames=sum(r["large_box_review"] for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(__file__).resolve().parent / "outputs/golf_club_first2")
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.run
    state = json.loads((root / "batch_status.json").read_text())
    if state["status"] != "completed" or not state["clips"]:
        raise ValueError("Run must be completed with at least one clip")
    out = root / "performance"
    out.mkdir(exist_ok=True)
    summaries, per_clip, frame_rows = [], [], []
    for clip in state["clips"]:
        records = [json.loads(line) for line in (root / Path(clip["source"]).stem / "detections.jsonl").read_text().splitlines()]
        records.sort(key=lambda r: r["frame"])
        if [r["frame"] for r in records] != list(range(clip["num_frames"])):
            raise ValueError(f"Missing/duplicate frames: {clip['source']}")
        if not np.isclose(sum(r["inference_seconds"] for r in records), clip["inference_seconds"], rtol=0, atol=1e-6):
            raise ValueError(f"Timing sum mismatch: {clip['source']}")
        summaries.append(summarize(clip["source"], records, len(records)/clip["fps"]))
        per_clip.append((clip["source"], records))
        for r in records:
            frame_rows.append(dict(video=clip["source"], frame=r["frame"], source_time_seconds=r["time_seconds"],
                                   inference_seconds=r["inference_seconds"],
                                   instantaneous_frames_per_second=1/r["inference_seconds"],
                                   valid_box_count=len(r["boxes"]), large_box_review=r["large_box_review"]))
    all_records = [r for _, records in per_clip for r in records]
    total = summarize("TOTAL", all_records, sum(s["source_duration_seconds"] for s in summaries))
    summaries.append(total)
    legacy = "backend" not in state and "backend" not in state["settings"]
    backend = state.get("backend", state["settings"].get("backend", "metal (legacy MLX run)"))
    hardware = state.get("hardware", "Apple M1 Pro, 16 GiB unified memory" if legacy else "Not recorded")
    notes = [
        "Timing includes image preparation, model generation and coordinate parsing. Loading, decoding/encoding video, previews, downtime and unsaved interrupted work are excluded. End-to-end wall time was not recorded.",
        "Throughput = frames / sum of saved inference seconds. Percentiles use NumPy linear interpolation; source duration = frames / source FPS. Compilation playback FPS is not inference FPS.",
        "Memory values describe the final process only, including model residency; they are not system RAM/RSS/swap or the peak of previous resumed processes. MLX peak and CUDA allocated/reserved peaks have different scopes.",
        "Valid boxes are parsed predictions, not measured accuracy. Large-box review means area >80% of the source image. No ground truth, IoU, precision/recall or mAP is available.",
        "Every source frame is independently inferred without tracking/interpolation. Different backends/checkpoint conversions may produce different predictions; this is not a controlled cross-device benchmark.",
    ]
    if legacy:
        notes.append("Historical first-two-video run: video 0 and video 1 frames 0-3 were saved before interruption; video 1 frames 4-323 ran after MLX wired limit 8 GiB/cache 256 MiB. Mixed-session timings are not a uniform benchmark.")
    memory = {k: v for k, v in state.items() if (k.endswith("_bytes") and "memory" in k) or k in {"cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"}}
    report = dict(model=state["settings"], backend=backend, hardware=hardware, timing_notes=notes,
                  memory=memory, summaries=summaries)
    if "mlx_peak_memory_bytes" in state:
        report.update(mlx_peak_memory_bytes=state["mlx_peak_memory_bytes"],
                      mlx_peak_memory_gib=state["mlx_peak_memory_bytes"]/1024**3)
    (out / "performance.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, rows in (("summary.csv", summaries), ("per_frame.csv", frame_rows)):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for name, records in per_clip:
        x = np.array([r["frame"] for r in records])
        y = np.array([r["inference_seconds"] for r in records])
        axes[0, 0].plot(x, y, linewidth=1, label=name)
        flags = np.array([r["large_box_review"] for r in records], dtype=bool)
        axes[0, 0].scatter(x[flags], y[flags], color="#ba2525", marker="x")
        axes[0, 1].hist(y, bins=30, alpha=.5, label=name)
    axes[0, 0].set(title="Frame latency (red crosses: large-box review)", xlabel="Source frame", ylabel="Seconds per frame")
    axes[0, 1].set(title="Latency distribution", xlabel="Seconds per frame", ylabel="Frame count", yscale="log")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    names = [s["video"] for s in summaries]
    fps = [s["throughput_frames_per_second"] for s in summaries]
    bars = axes[1, 0].bar(names, fps, color="#247ba0")
    axes[1, 0].bar_label(bars, labels=[f"{v:.3f}" for v in fps], padding=3)
    axes[1, 0].margins(y=.15)
    axes[1, 0].set(title="Measured inference throughput", ylabel="Frames per second")
    axes[1, 1].bar(names, [s["inference_seconds"] for s in summaries], color="#4b8262")
    axes[1, 1].set(title="Saved inference time", ylabel="Seconds")
    for ax in axes[1]:
        ax.tick_params(axis="x", labelrotation=30)
    fig.suptitle(f"LocateAnything / {backend} / {hardware}\nSaved frame timings; loading, restarts and video export excluded", fontsize=12)
    fig.savefig(out / "performance.png", dpi=180)
    fig.savefig(out / "performance.pdf")
    plt.close(fig)

    headers = ["Video", "Frames", "Inference seconds", "Mean s/frame", "Median", "P95", "Inference FPS", "Slower than realtime", "Review frames"]
    rows = [[s["video"], str(s["frames"]), f"{s['inference_seconds']:.2f}", f"{s['mean_seconds_per_frame']:.3f}",
             f"{s['median_seconds_per_frame']:.3f}", f"{s['p95_seconds_per_frame']:.3f}",
             f"{s['throughput_frames_per_second']:.3f}", f"{s['slower_than_realtime']:.1f}x", str(s['large_box_review_frames'])] for s in summaries]
    intro = (f"{len(per_clip)} video, {total['frames']} frame; backend {backend}; hardware {hardware}. "
             f"Saved inference: {total['inference_seconds']:.2f} seconds ({total['inference_seconds']/60:.2f} minutes).")
    memory_text = "; ".join(f"{k}: {v/1024**3:.2f} GiB" for k, v in memory.items()) or "Memory not recorded"
    table = ["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)]
    table += ["| " + " | ".join(row) + " |" for row in rows]
    (out / "REPORT.md").write_text("# VLM performance\n\n" + intro + "\n\n" + memory_text + "\n\n" + "\n".join(table) + "\n\n" + "\n".join("- " + n for n in notes) + "\n\n![Chart](performance.png)\n")
    def cells(row, tag):
        return "<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in row) + "</tr>"
    b64 = base64.b64encode((out / "performance.png").read_bytes()).decode()
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>VLM performance</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 20px}'
                'table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #ddd}img{width:100%}</style>'
                '<h1>VLM performance</h1><p>' + html.escape(intro) + '</p><p>' + html.escape(memory_text) + '</p>'
                '<div style="overflow-x:auto"><table><thead>' + cells(headers, "th") + '</thead><tbody>'
                + ''.join(cells(row, "td") for row in rows) + '</tbody></table></div>'
                '<img alt="Measured inference performance" src="data:image/png;base64,' + b64 + '"><ul>'
                + ''.join('<li>'+html.escape(n)+'</li>' for n in notes) + '</ul></html>')
    (out / "report.html").write_text(document)
    print(json.dumps({"output": str(out.resolve()), "summary": total, "memory": memory}, indent=2))


if __name__ == "__main__":
    main()
