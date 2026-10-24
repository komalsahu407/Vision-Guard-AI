"""Behaviour analysis: tracks -> incidents -> evidence + baseline comparison.

What is real here
  * Detection/tracking come from pipeline.py (YOLOv8n + IoU tracker).
  * Incident rules (zone, loitering, unattended object, rapid movement) are transparent rules.
  * Anomaly score is an Isolation Forest fitted on THIS video's person tracks. It switches
    itself off when there are too few tracks (config.ANOMALY_MIN_TRACKS) instead of faking a score.
  * Severity = sum of named factor points, so the explanation IS the calculation.
"""
from __future__ import annotations

import bisect
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
from core.pipeline import cache_dir_for

KIND_LABELS = {
    "loiter": "Loitering in restricted zone",
    "abandoned": "Unattended object",
    "running": "Rapid movement",
    "anomaly": "Unusual movement pattern",
    "entry": "Entered restricted zone",
}
ENTRY_MIN_S = 1.0     # seconds inside the zone before "entry" mode counts a visit

DEFAULT_PARAMS = {
    "loiter_s": config.LOITER_S,
    "abandon_s": config.ABANDON_S,
    "run_speed": config.RUN_SPEED,
    "min_severity": config.MIN_SEVERITY,
    "min_track_s": config.MIN_TRACK_S,
    "entry_alerts": 0,        # 1 = also raise a (low severity) alert for every person who really entered the zone
    **{"w_" + k: v for k, v in config.WEIGHTS.items()},     # editable severity weights (Alerts page)
}
_ACTIVE_W = dict(config.WEIGHTS)


# ---------------------------------------------------------------- helpers
def merge_params(p: dict | None) -> dict:
    out = dict(DEFAULT_PARAMS)
    for k, v in (p or {}).items():
        if k in out:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _inside(poly: np.ndarray | None, pt) -> bool:
    if poly is None:
        return False
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def _steps(t: np.ndarray) -> np.ndarray:
    """Seconds each sample 'covers' (gaps after a lost track are capped at 1 s)."""
    if len(t) < 2:
        return np.zeros(len(t))
    d = np.diff(t)
    return np.minimum(np.append(d, np.median(d)), 1.0)


def _speed(t: np.ndarray, pts: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Speed in body-heights per second, measured over ~1 s windows (robust to detection jitter)."""
    n = len(t)
    sp = np.zeros(n)
    j = 0
    for i in range(n):
        j = max(j, i)
        while j < n - 1 and t[j] - t[i] < 1.0:
            j += 1
        dt = t[j] - t[i]
        if dt >= 0.5:
            sp[i] = np.hypot(*(pts[j] - pts[i])) / max(float(np.mean(h[i:j + 1])), 1.0) / dt
        elif i > 0:
            sp[i] = sp[i - 1]
    return sp


def _interval(t: np.ndarray, mask: np.ndarray):
    if not mask.any():
        return None
    return float(t[mask][0]), float(t[mask][-1])


def _box_at(tr: dict, f: int, stride: int):
    frames = tr["frames"]
    i = bisect.bisect_left(frames, f)
    best = None
    for k in (i - 1, i):
        if 0 <= k < len(frames) and abs(frames[k] - f) <= stride:
            if best is None or abs(frames[k] - f) < abs(frames[best] - f):
                best = k
    return None if best is None else tr["boxes"][best]


def _level(sev: float) -> str:
    return "high" if sev >= 70 else "medium" if sev >= 45 else "low"


def _factor(key, label, value, detail):
    w = _ACTIVE_W.get(key, config.WEIGHTS[key])
    value = float(max(0.0, min(1.0, value)))
    return {"key": key, "label": label, "weight": w, "value": round(value, 3),
            "points": round(w * value, 1), "detail": detail}


def _stitch(tracks: list[dict], fps: float, max_gap_s: float = 2.0) -> list[dict]:
    """Join person tracks that broke apart (occlusion, missed detections) into one identity.
    A track is attached to an earlier one if it starts within max_gap_s and close to where that one ended."""
    people = sorted([t for t in tracks if t["label"] in config.PERSON_LABELS], key=lambda t: t["frames"][0])
    others = [t for t in tracks if t["label"] not in config.PERSON_LABELS]
    merged: list[dict] = []
    for t in people:
        best = None
        for m in merged:
            gap = (t["frames"][0] - m["frames"][-1]) / fps
            if 0 < gap <= max_gap_s:
                h = float(np.mean([b[3] - b[1] for b in m["boxes"][-5:]]))
                d = float(np.hypot(t["path"][0][0] - m["path"][-1][0], t["path"][0][1] - m["path"][-1][1]))
                if d < 0.8 * max(h, 20) and (best is None or d < best[0]):
                    best = (d, m)
        if best:
            m = best[1]
            m["frames"] += t["frames"]; m["path"] += t["path"]; m["boxes"] += t["boxes"]; m["ids"].append(t["id"])
        else:
            merged.append({**t, "frames": list(t["frames"]), "path": list(t["path"]),
                           "boxes": list(t["boxes"]), "ids": [t["id"]]})
    return merged + others


# ---------------------------------------------------------------- per-track stats
def _person_stats(tr: dict, fps: float, poly, prm: dict) -> dict:
    t = np.array(tr["frames"], float) / fps
    pts = np.array(tr["path"], float)
    bx = np.array(tr["boxes"], float)
    h = np.maximum(bx[:, 3] - bx[:, 1], 1.0)
    sp = _speed(t, pts, h)
    step = _steps(t)
    inz = np.array([_inside(poly, q) for q in pts], bool)
    still = sp < config.STILL_SPEED
    path_len = float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 0.0
    net = float(np.hypot(*(pts[-1] - pts[0]))) if len(pts) > 1 else 0.0
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    return {
        "t": t, "sp": sp, "inz": inz, "step": step,
        "in_zone_s": float(step[inz].sum()),
        "still_s": float(step[still].sum()),
        "p90": float(np.percentile(sp, 90)) if len(sp) else 0.0,
        "run_iv": _interval(t, sp > prm["run_speed"]),
        "zone_iv": _interval(t, inz),
        "features": [float(np.median(sp)), float(np.percentile(sp, 90)),
                     net / path_len if path_len > 1 else 1.0, dur,
                     float(step[still].sum() / max(step.sum(), 1e-6)),
                     float(step[inz].sum() / max(step.sum(), 1e-6))],
    }


def _anomaly_scores(person_ids: list[int], feats: list[list[float]]):
    """Isolation Forest on this video's person tracks. Returns (dict id->0..1, status text)."""
    n = len(person_ids)
    if n < config.ANOMALY_MIN_TRACKS:
        return {}, f"inactive: needs at least {config.ANOMALY_MIN_TRACKS} person tracks, found {n}"
    from sklearn.ensemble import IsolationForest
    x = np.array(feats, float)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    model = IsolationForest(n_estimators=200, random_state=0).fit(x)
    s = -model.score_samples(x)              # ~0.5 normal, towards 1 = more unusual
    vals = np.clip((s - 0.5) / 0.15, 0, 1)
    return {pid: float(v) for pid, v in zip(person_ids, vals)}, f"active: fitted on {n} person tracks"


# ---------------------------------------------------------------- evidence images
def _render_evidence(video_path: Path, tracks: list[dict], target_ids: list[int], frame_idx: int,
                     meta: dict, out_dir: Path, name: str, caption: str):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    pw, ph = meta["proc_size"]
    frame = cv2.resize(frame, (pw, ph))
    stride = meta["stride"]
    raw, blur = frame.copy(), frame.copy()
    for tr in tracks:
        b = _box_at(tr, frame_idx, stride)
        if b is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(pw, x2), min(ph, y2)
        if tr["label"] in config.PERSON_LABELS and x2 > x1 and y2 > y1:
            roi = blur[y1:y2, x1:x2]
            k = max(15, ((x2 - x1) // 2) * 2 + 1)
            blur[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    for img in (raw, blur):
        for tr in tracks:
            if tr["id"] not in target_ids:
                continue
            b = _box_at(tr, frame_idx, stride)
            if b is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 90, 255), 3)
        cv2.rectangle(img, (0, 0), (pw, 26), (20, 14, 8), -1)
        cv2.putText(img, caption, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 120), 1, cv2.LINE_AA)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{name}.jpg"), blur, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(out_dir / f"{name}_raw.jpg"), raw, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


# ---------------------------------------------------------------- ground-truth evaluation
def _overlap(a0, a1, b0, b1, tol=1.0):
    return a0 <= b1 + tol and b0 <= a1 + tol


def _evaluate(gt: dict, incidents: list[dict], alert_times: list[float]):
    events = gt.get("events", [])
    if not events:
        return None
    matched_inc, matched_gt = set(), set()
    for gi, g in enumerate(events):
        for ii, inc in enumerate(incidents):
            if g.get("type") in inc["kinds"] and _overlap(inc["start_t"], inc["end_t"], g["start"], g["end"]):
                matched_inc.add(ii)
                matched_gt.add(gi)
    n_alerts = len(alert_times)
    good_alerts = sum(1 for a in alert_times if any(g["start"] - 1 <= a <= g["end"] + 1 for g in events))
    naive_gt = sum(1 for g in events if any(g["start"] - 1 <= a <= g["end"] + 1 for a in alert_times))
    div = lambda a, b: round(a / b, 3) if b else None
    return {
        "gt_events": len(events),
        "ours": {"alerts": len(incidents), "precision": div(len(matched_inc), len(incidents)),
                 "recall": div(len(matched_gt), len(events))},
        "baseline": {"alerts": n_alerts, "precision": div(good_alerts, n_alerts),
                     "recall": div(naive_gt, len(events))},
    }


# ---------------------------------------------------------------- main entry
def analyze(stem: str, params: dict | None = None) -> dict:
    cdir = cache_dir_for(Path(stem))
    tracks_file = cdir / "tracks.json"
    if not tracks_file.exists():
        raise FileNotFoundError("Run detection first (tracks.json missing).")
    data = json.loads(tracks_file.read_text())
    meta, raw_tracks = data["meta"], data["tracks"]
    if raw_tracks and "boxes" not in raw_tracks[0]:
        raise RuntimeError("Old cache without boxes. Re-run detection with --force.")
    prm = merge_params(params)
    _ACTIVE_W.update({k: max(0.0, min(100.0, prm["w_" + k])) for k in config.WEIGHTS})
    fps = meta["src_fps"]
    tracks = [t for t in _stitch(raw_tracks, fps)
              if (t["frames"][-1] - t["frames"][0]) / fps >= prm["min_track_s"]]   # drop short flicker tracks
    kept_ids = sorted({i for t in tracks for i in t.get("ids", [t["id"]])})
    stride = meta["stride"]

    zone_file = cdir / "zone.json"
    zone = json.loads(zone_file.read_text()).get("points") if zone_file.exists() else None
    poly = np.array(zone, np.float32) if zone and len(zone) >= 3 else None

    persons = [t for t in tracks if t["label"] in config.PERSON_LABELS]
    bags = [t for t in tracks if t["label"] in config.BAG_LABELS]
    stats = {t["id"]: _person_stats(t, fps, poly, prm) for t in persons}
    anomaly, anomaly_status = _anomaly_scores(list(stats), [s["features"] for s in stats.values()])

    incidents = []

    # ---- people
    for tr in persons:
        s = stats[tr["id"]]
        kinds, ivs, notes = [], [], []
        loiter = poly is not None and s["in_zone_s"] >= prm["loiter_s"]
        if loiter:
            kinds.append("loiter"); ivs.append(s["zone_iv"])
        entry = (not loiter) and prm["entry_alerts"] >= 0.5 and poly is not None and s["in_zone_s"] >= ENTRY_MIN_S and s["zone_iv"] is not None
        if entry:
            kinds.append("entry"); ivs.append(s["zone_iv"])
        run = s["p90"] > prm["run_speed"] and s["run_iv"] is not None
        if run:
            kinds.append("running"); ivs.append(s["run_iv"])
        a = anomaly.get(tr["id"], 0.0)
        if not kinds:
            continue                      # an incident needs at least one rule trigger; anomaly alone never raises one
        if a >= 0.6:
            kinds.append("anomaly")       # shown as supporting evidence
        alert_ts = []
        if loiter:
            cum = np.cumsum(np.where(s["inz"], s["step"], 0.0))
            alert_ts.append(float(s["t"][int(np.argmax(cum >= prm["loiter_s"]))]))
        if entry:
            alert_ts.append(float(s["t"][int(np.argmax(s["inz"]))]))
        if run:
            alert_ts.append(float(s["t"][int(np.argmax(s["sp"] > prm["run_speed"]))]))
        f_zone = _factor("zone_entry", "Entered restricted zone", 1.0 if s["in_zone_s"] > 0 else 0.0,
                         f"{s['in_zone_s']:.1f} s inside the zone" if poly is not None else "no zone drawn")
        f_dwell = _factor("dwell", "Time spent in zone", s["in_zone_s"] / (2 * prm["loiter_s"]),
                          f"{s['in_zone_s']:.1f} s of {2 * prm['loiter_s']:.0f} s scale (loiter limit {prm['loiter_s']:.0f} s)")
        f_run = _factor("running", "Rapid movement",
                        0.5 + 0.5 * min(1.0, (s["p90"] - prm["run_speed"]) / prm["run_speed"]) if run else 0.0,
                        f"speed {s['p90']:.2f} body-heights/s (limit {prm['run_speed']:.1f})")
        f_anom = _factor("anomaly", "Unusual vs this scene", a,
                         anomaly_status.split(":")[0] if not anomaly else f"Isolation Forest score {a:.2f}")
        factors = [f_zone, f_dwell, f_run, f_anom]
        sev = min(100.0, sum(f["points"] for f in factors))
        if sev < prm["min_severity"]:
            continue
        start = min(i[0] for i in ivs)
        end = max(i[1] for i in ivs)
        parts = []
        if loiter:
            parts.append(f"stayed {s['in_zone_s']:.1f} s in the restricted zone (limit {prm['loiter_s']:.0f} s)")
        if entry:
            parts.append(f"entered the restricted zone and stayed {s['in_zone_s']:.1f} s (below the {prm['loiter_s']:.0f} s loiter limit)")
        if run:
            parts.append(f"moved at {s['p90']:.1f} body-heights/s (limit {prm['run_speed']:.1f})")
        if "anomaly" in kinds:
            parts.append("its movement differs from the other tracks in this scene")
        incidents.append({"track_id": tr["id"], "track_ids": tr.get("ids", [tr["id"]]), "label": tr["label"], "kinds": kinds, "start_t": round(start, 2),
                          "end_t": round(end, 2), "alert_t": round(min(alert_ts), 2), "severity": round(sev, 1), "level": _level(sev),
                          "factors": factors,
                          "summary": f"{tr['label'].capitalize()} #{tr['id']} " + ("; ".join(parts) if parts else "scored above the alert threshold") + "."})

    # ---- unattended objects
    ref_h = float(np.median([np.mean([b[3] - b[1] for b in t["boxes"]]) for t in persons])) if persons else 150.0
    at = defaultdict(list)
    for t in persons:
        for f, p in zip(t["frames"], t["path"]):
            at[f].append(p)
    for tr in bags:
        t = np.array(tr["frames"], float) / fps
        pts = np.array(tr["path"], float)
        if len(t) < 3:
            continue
        bh = float(np.mean([b[3] - b[1] for b in tr["boxes"]]))
        still = float(np.max(np.hypot(*(pts - np.median(pts, axis=0)).T))) < 0.6 * max(bh, 20)
        if not still:
            continue
        step = _steps(t)
        unatt = np.zeros(len(t), bool)
        for i, (f, p) in enumerate(zip(tr["frames"], pts)):
            near = [q for off in (-stride, 0, stride) for q in at.get(f + off, [])]
            unatt[i] = not near or min(np.hypot(q[0] - p[0], q[1] - p[1]) for q in near) > ref_h
        un_s = float(step[unatt].sum())
        if un_s < prm["abandon_s"]:
            continue
        iv = (float(t[unatt][0]), float(t[unatt][-1]))
        cum = np.cumsum(np.where(unatt, step, 0.0))
        bag_alert = float(t[int(np.argmax(cum >= prm["abandon_s"]))])
        in_zone = poly is not None and any(_inside(poly, q) for q in pts)
        factors = [_factor("abandoned", "Unattended object", min(1.0, un_s / prm["abandon_s"]),
                           f"still and nobody within {ref_h:.0f}px for {un_s:.1f} s (limit {prm['abandon_s']:.0f} s)"),
                   _factor("zone_entry", "Inside restricted zone", 1.0 if in_zone else 0.0,
                           "object lies in the zone" if in_zone else "object outside zone")]
        sev = min(100.0, sum(f["points"] for f in factors))
        incidents.append({"track_id": tr["id"], "track_ids": [tr["id"]], "label": tr["label"], "kinds": ["abandoned"],
                          "start_t": round(iv[0], 2), "end_t": round(iv[1], 2), "alert_t": round(bag_alert, 2), "severity": round(sev, 1),
                          "level": _level(sev), "factors": factors,
                          "summary": f"{tr['label'].capitalize()} #{tr['id']} was left alone for {un_s:.1f} s."})

    incidents.sort(key=lambda i: -i["severity"])
    video_path = config.VIDEO_DIR / meta["video"]
    ev_dir = cdir / "evidence"
    for n, inc in enumerate(incidents, 1):
        inc["id"] = n
        inc["type"] = " + ".join(KIND_LABELS[k] for k in inc["kinds"]) or "Elevated risk score"
        mid = (inc["start_t"] + inc["end_t"]) / 2
        inc["peak_t"] = round(mid, 2)
        f_idx = int(round(mid * fps / stride) * stride)
        cap = f"INCIDENT {n} | {inc['type']} | severity {inc['severity']:.0f} | t={mid:.1f}s"
        inc["evidence"] = bool(video_path.exists() and _render_evidence(
            video_path, raw_tracks, inc["track_ids"], f_idx, meta, ev_dir, f"inc{n}", cap))

    # ---- baseline: naive rule = alert on every processed frame that has a person inside the zone
    alert_frames = sorted({f for t in persons for f, p in zip(t["frames"], t["path"]) if _inside(poly, p)})
    alert_times = [f / fps for f in alert_frames]
    naive = len(alert_frames)
    zone_visitors = sum(1 for s in stats.values() if s["in_zone_s"] > 0)
    flagged = {i["track_id"] for i in incidents}
    gt_file = config.VIDEO_DIR / (Path(meta["video"]).stem + ".gt.json")
    evaluation = _evaluate(json.loads(gt_file.read_text()), incidents, alert_times) if gt_file.exists() else None

    result = {
        "video": meta["video"], "params": prm, "zone": zone, "meta": meta,
        "version": 3, "kept_ids": kept_ids, "ignored_tracks": len(raw_tracks) - len(kept_ids),
        "identity": {str(i): t["id"] for t in tracks for i in t.get("ids", [t["id"]])},
        "anomaly_status": anomaly_status, "incidents": incidents, "evaluation": evaluation,
        "naive_times": [round(a, 2) for a in alert_times] if poly is not None else [],
        "kpis": {
            "tracks": len(tracks), "persons": len(persons), "bags": len(bags), "incidents": len(incidents),
            "naive_alerts": naive if poly is not None else None,
            "reduction_pct": round(100 * (1 - len(incidents) / naive), 1) if poly is not None and naive else None,
            "zone_visitors": zone_visitors if poly is not None else None,
            "visitors_not_flagged": sum(1 for pid, s in stats.items() if s["in_zone_s"] > 0 and pid not in flagged)
            if poly is not None else None,
            "avg_infer_ms": meta["avg_infer_ms"], "processing_fps": meta["processing_fps"],
        },
    }
    (cdir / "analysis.json").write_text(json.dumps(result))
    return result
