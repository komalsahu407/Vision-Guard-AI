"""Does the YOLO model 'see' a bag in one exact frame?  (no filtering, so you see the raw scores)

Run:   python diagnose_bag.py 05.avi 29.8
Prints the best backpack / handbag / suitcase / person scores for that moment, for every model in models/.
Saves diag_<time>.jpg with the strongest bag candidate drawn, so you can look at it.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from core import detector as D

if len(sys.argv) < 3:
    sys.exit("Usage: python diagnose_bag.py <video file name in data\\videos> <time in seconds>")
vp = config.VIDEO_DIR / sys.argv[1]
t = float(sys.argv[2])
cap = cv2.VideoCapture(str(vp))
if not cap.isOpened():
    sys.exit(f"Cannot open {vp}")
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
best_img = None
frames = []
for dt in (-0.2, -0.1, 0, 0.1, 0.2):          # a few frames around the moment
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((t + dt) * fps)))
    ok, fr = cap.read()
    if ok:
        h, w = fr.shape[:2]
        frames.append(cv2.resize(fr, (config.PROC_WIDTH, int(round(h * config.PROC_WIDTH / w / 2) * 2))))
cap.release()
if not frames:
    sys.exit("Could not read frames at that time.")

import onnxruntime as ort
models = [p for p in (config.MODEL_PATH, config.MODEL_PATH_640) if p.exists()]
if not models:
    sys.exit("No model found in models/")
ALL = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}
for mp in models:
    sess = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = D.model_input_size(inp.shape, config.INPUT_SIZE)
    best = {}
    for fr in frames:
        h, w = fr.shape[:2]
        sc = size / max(h, w)
        nw, nh = int(round(w * sc)), int(round(h * sc))
        canvas = np.full((size, size, 3), 114, np.uint8)
        px, py = (size - nw) // 2, (size - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(fr, (nw, nh))
        blob = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        out = sess.run(None, {inp.name: blob})[0]
        for d in D.decode_yolov8(out, sc, px, py, 0.02, config.NMS_THRESHOLD, ALL, w, h):
            if d["label"] not in best or d["conf"] > best[d["label"]][0]:
                best[d["label"]] = (d["conf"], d["box"], fr)
    print(f"\n=== {mp.name}  (input {size}px) ===")
    for lab in ("person", "backpack", "handbag", "suitcase"):
        if lab in best:
            print(f"  {lab:9s} best score {best[lab][0]:.2f}")
        else:
            print(f"  {lab:9s} not seen at all (score below 0.02)")
    bags = [(v[0], k, v) for k, v in best.items() if k != "person"]
    if bags:
        _, lab, (cf, box, fr) = max(bags)
        img = fr.copy()
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(img, f"{lab} {cf:.2f}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        out_name = f"diag_{mp.stem}_{t}.jpg"
        cv2.imwrite(out_name, img)
        print(f"  saved {out_name}")
print(f"\nBag threshold in config.py is {config.BAG_CONF_THRESHOLD}. A bag score below that is never shown.")
