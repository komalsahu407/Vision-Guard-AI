"""Video -> detections -> tracks -> cached results (tracks.json + preview.mp4).

Everything the dashboard later shows is computed here from real frames.
Results are cached so the demo replays instantly even on a slow laptop.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import cv2

import config
from core.detector import load_detector, signature
from core.tracker import Tracker

COLORS = {"person": (255, 200, 0), "backpack": (0, 165, 255), "handbag": (0, 165, 255),
          "suitcase": (0, 165, 255), "motion": (200, 120, 255)}


def cache_dir_for(video_path: Path) -> Path:
    return config.CACHE_DIR / video_path.stem


def run(video_path: Path, prefer_yolo: bool = True, force: bool = False,
        on_progress: Callable[[float, str], None] | None = None) -> dict:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = cache_dir_for(video_path)
    meta_file = out_dir / "meta.json"
    if meta_file.exists() and not force:
        return json.loads(meta_file.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}. Try an .mp4 (H.264).")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pw = config.PROC_WIDTH
    ph = int(round(sh * pw / sw / 2) * 2)

    detector, warning = load_detector(prefer_yolo)
    tracker = Tracker()
    writer = cv2.VideoWriter(str(out_dir / "preview.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             src_fps / config.FRAME_STRIDE, (pw, ph))

    frames_log = []          # per processed frame: time + detections
    infer_times = []
    idx = -1
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % config.FRAME_STRIDE:
                continue
            frame = cv2.resize(frame, (pw, ph))

            t1 = time.time()
            dets = detector.detect(frame)
            infer_times.append(time.time() - t1)

            confirmed = tracker.step(dets, idx)
            for t in confirmed:
                x1, y1, x2, y2 = map(int, t.box)
                c = COLORS.get(t.label, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                cv2.putText(frame, f"{t.label} #{t.id} {t.conf:.2f}", (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
            writer.write(frame)

            frames_log.append({"f": idx, "t": round(idx / src_fps, 3),
                               "n": len(confirmed),
                               "b": [[t.id, t.label] + [int(v) for v in t.box]
                                     for t in confirmed]})
            if on_progress and total and len(frames_log) % 10 == 0:
                on_progress(min(idx / total, 1.0), f"frame {idx}/{total}")
    finally:
        cap.release()
        writer.release()

    tracks = [{
        "id": t.id, "label": t.label,
        "first_frame": t.first_frame, "last_frame": t.last_frame, "hits": t.hits,
        "start_t": round(t.first_frame / src_fps, 3), "end_t": round(t.last_frame / src_fps, 3),
        "frames": t.frames, "boxes": [[int(v) for v in b] for b in t.boxes], "path": [[round(x, 1), round(y, 1)] for x, y in t.path],
    } for t in tracker.all_tracks()]

    wall = time.time() - t0
    meta = {
        "video": video_path.name, "detector": detector.name, "warning": warning, "sig": signature(),
        "src_fps": round(src_fps, 2), "proc_size": [pw, ph], "stride": config.FRAME_STRIDE,
        "processed_frames": len(frames_log), "total_frames": idx + 1,
        "n_tracks": len(tracks),
        "avg_infer_ms": round(1000 * sum(infer_times) / max(len(infer_times), 1), 1),
        "processing_fps": round(len(frames_log) / max(wall, 1e-6), 1),
        "wall_seconds": round(wall, 1),
    }
    (out_dir / "tracks.json").write_text(json.dumps({"meta": meta, "tracks": tracks,
                                                     "frames": frames_log}))
    meta_file.write_text(json.dumps(meta, indent=2))
    if on_progress:
        on_progress(1.0, "done")
    return meta
