"""Create a synthetic test clip (moving blocks on a plain scene) to test the
pipeline without real footage. Blocks are NOT people, so use --no-yolo with it.

    python make_test_video.py     ->  data/videos/synthetic.mp4
"""
import cv2
import numpy as np

import config


def main():
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out = config.VIDEO_DIR / "synthetic.mp4"
    w, h, fps, seconds = 640, 360, 25, 12
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    bg = np.full((h, w, 3), 40, np.uint8)
    cv2.rectangle(bg, (0, 280), (w, h), (55, 55, 55), -1)

    for f in range(fps * seconds):
        frame = bg.copy()
        frame = np.clip(frame + rng.integers(-3, 4, frame.shape), 0, 255).astype(np.uint8)
        # A: walks straight across the whole clip
        ax = int(30 + f * 2.2)
        cv2.rectangle(frame, (ax, 190), (ax + 40, 290), (200, 200, 220), -1)
        # B: appears at 4s, stands still (loiters) until the end
        if f > fps * 4:
            sway = int(14 * np.sin(f / 8.0))
            cv2.rectangle(frame, (440 + sway, 200), (480 + sway, 300), (180, 220, 180), -1)
        # C: crosses quickly between 7s and 9s
        if fps * 7 < f < fps * 9:
            cx = int(600 - (f - fps * 7) * 8)
            cv2.rectangle(frame, (cx, 210), (cx + 36, 295), (160, 160, 230), -1)
        vw.write(frame)
    vw.release()
    print("Wrote", out)


if __name__ == "__main__":
    main()
