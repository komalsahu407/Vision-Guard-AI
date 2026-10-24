
"""VisionGuard AI: explainable rule-based alerts."""
import json
import math
from pathlib import Path


def analyze(tracks_file):
    tracks_file = Path(tracks_file)
    data = json.loads(tracks_file.read_text(encoding="utf-8"))

    meta = data["meta"]
    fps = max(float(meta.get("src_fps", 25)), 1.0)
    width, height = meta["proc_size"]

    # Configurable demo zone; not a real-world danger zone.
    zone = (
        width * 0.35, height * 0.25,
        width * 0.65, height * 0.85
    )

    alerts = []

    def add_alert(kind, track_id, severity, frame, reason, message):
        alerts.append({
            "type": kind,
            "track_id": track_id,
            "severity": severity,
            "timestamp_seconds": round(frame / fps, 2),
            "evidence_frame": frame,
            "reason": reason,
            "message": message
        })

    for track in data.get("tracks", []):
        label = track.get("label", "")
        track_id = track.get("id")
        points = track.get("path", [])
        frames = track.get("frames", [])

        if not points or not frames or len(points) != len(frames):
            continue

        duration = max(0, (frames[-1] - frames[0]) / fps)

        # Restricted-zone rule: first observed point inside the zone.
        if label == "person":
            for (x, y), frame in zip(points, frames):
                inside = (
                    zone[0] <= x <= zone[2]
                    and zone[1] <= y <= zone[3]
                )
                if inside:
                    add_alert(
                        "restricted_zone", track_id, "high", frame,
                        "Person track point was observed inside the configured demo zone.",
                        "Demo-zone presence detected; human verification required."
                    )
                    break

        # Loitering heuristic: low positional variation over time.
        if label == "person" and duration >= 20:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            max_deviation = max(
                math.dist((p[0], p[1]), (cx, cy)) for p in points
            )

            if max_deviation <= 50:
                add_alert(
                    "possible_loitering", track_id, "medium", frames[-1],
                    "Track lasted at least 20 seconds with limited positional variation.",
                    "Possible loitering; review the video before drawing conclusions."
                )

        # Stationary-bag heuristic, not proof of abandonment.
        if label in ("backpack", "handbag", "suitcase") and duration >= 15:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            max_deviation = max(
                math.dist((p[0], p[1]), (cx, cy)) for p in points
            )

            if max_deviation <= 35:
                add_alert(
                    "stationary_bag", track_id, "medium", frames[-1],
                    "Bag track lasted at least 15 seconds with limited positional variation.",
                    "Stationary bag detected; human verification required."
                )

    output = tracks_file.parent / "alerts.json"
    output.write_text(json.dumps(alerts, indent=2), encoding="utf-8")
    return alerts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tracks_file")
    args = parser.parse_args()

    results = analyze(args.tracks_file)
    print(json.dumps(results, indent=2))
    print(f"\nAlerts saved to: {Path(args.tracks_file).parent / 'alerts.json'}")
