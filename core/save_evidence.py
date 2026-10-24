
"""Extract video evidence frames for VisionGuard alerts."""
import json
from pathlib import Path
import cv2

VIDEO = Path("data/videos/vision.mp4")
ALERTS = Path("data/cache/vision/alerts.json")
OUTPUT = Path("data/cache/vision/evidence")


def main():
    if not VIDEO.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO}")
    if not ALERTS.exists():
        raise FileNotFoundError(f"Alerts not found: {ALERTS}")

    alerts = json.loads(ALERTS.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    saved = 0

    try:
        for alert in alerts:
            frame_number = int(alert["evidence_frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()

            if not ok:
                print(f"Could not read frame {frame_number}")
                continue

            filename = (
                f"alert_{alert['type']}_"
                f"track_{alert['track_id']}_frame_{frame_number}.jpg"
            )
            image_path = OUTPUT / filename

            if cv2.imwrite(str(image_path), frame):
                alert["evidence_image"] = str(image_path).replace("\\", "/")
                saved += 1
                print(f"Saved: {image_path}")
            else:
                print(f"Could not save image: {image_path}")
    finally:
        cap.release()

    ALERTS.write_text(
        json.dumps(alerts, indent=2),
        encoding="utf-8"
    )
    print(f"\nSaved {saved} evidence image(s) to: {OUTPUT}")


if __name__ == "__main__":
    main()
