"""Detectors.

YoloOnnxDetector  - real pretrained YOLOv8n (ONNX) running on CPU via onnxruntime.
MotionDetector    - fallback that finds moving blobs (background subtraction).
                    It is NOT an object classifier: every blob is labelled "motion".
                    Use it only to test the pipeline when the ONNX model is missing.

Both return a list of dicts: {"box": [x1, y1, x2, y2], "conf": float, "label": str}
"""
from __future__ import annotations

import cv2
import numpy as np

import config


def decode_yolov8(output: np.ndarray, scale: float, pad_x: float, pad_y: float,
                  conf_thr: float, nms_thr: float, classes: dict[int, str],
                  frame_w: int, frame_h: int) -> list[dict]:
    """Turn raw YOLOv8 output of shape (1, 4+nc, N) into detections in frame pixels."""
    preds = output[0].T                      # (N, 4+nc)
    boxes_xywh = preds[:, :4]
    scores = preds[:, 4:]

    class_ids = scores.argmax(axis=1)
    confs = scores[np.arange(len(scores)), class_ids]

    # people use conf_thr; bags use the lower BAG_CONF_THRESHOLD
    thr = np.where(class_ids == 0, conf_thr, min(conf_thr, config.BAG_CONF_THRESHOLD))
    keep = np.isin(class_ids, list(classes.keys())) & (confs >= thr)
    if not keep.any():
        return []
    boxes_xywh, class_ids, confs = boxes_xywh[keep], class_ids[keep], confs[keep]

    # centre-xywh (in letterboxed input space) -> top-left xywh in original frame
    cx, cy, w, h = boxes_xywh.T
    x = (cx - w / 2 - pad_x) / scale
    y = (cy - h / 2 - pad_y) / scale
    w = w / scale
    h = h / scale

    idx = cv2.dnn.NMSBoxes(
        np.stack([x, y, w, h], axis=1).tolist(), confs.tolist(), min(conf_thr, config.BAG_CONF_THRESHOLD), nms_thr
    )
    idx = np.array(idx).flatten() if len(idx) else []

    out = []
    for i in idx:
        x1 = float(np.clip(x[i], 0, frame_w - 1))
        y1 = float(np.clip(y[i], 0, frame_h - 1))
        x2 = float(np.clip(x[i] + w[i], 0, frame_w - 1))
        y2 = float(np.clip(y[i] + h[i], 0, frame_h - 1))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        if classes[int(class_ids[i])] == "person" and (y2 - y1) < config.MIN_PERSON_H:
            continue
        out.append({"box": [x1, y1, x2, y2], "conf": float(confs[i]),
                    "label": classes[int(class_ids[i])]})
    return out


def resolve_model_path():
    """Prefer the 640 model if the user exported it, else the default 320 one."""
    return config.MODEL_PATH_640 if config.MODEL_PATH_640.exists() else config.MODEL_PATH


def model_input_size(shape, default: int) -> int:
    """Read the square input size from the ONNX input shape (e.g. [1,3,640,640]); fall back to default."""
    try:
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int) and h == w and h > 0:
            return h
    except (TypeError, IndexError):
        pass
    return default


def signature() -> str:
    """Changes whenever the model or any detection setting changes, so old cached results are redone."""
    p = resolve_model_path()
    model = f"{p.name}:{p.stat().st_size}" if p.exists() else "none"
    return "|".join(str(x) for x in (model, config.CONF_THRESHOLD, config.BAG_CONF_THRESHOLD, config.TRACK_MIN_HITS_BAG, config.BAG_SOLO_CONF, 'v2', config.NMS_THRESHOLD, config.TRACK_MIN_HITS,
                                     config.TRACK_MAX_AGE, config.TRACK_IOU_MIN, config.MIN_PERSON_H,
                                     config.PROC_WIDTH, config.FRAME_STRIDE))


class YoloOnnxDetector:
    def __init__(self, model_path=None, size: int | None = None):
        import onnxruntime as ort  # imported here so the fallback works without it

        model_path = model_path or resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Export it in Colab "
                f"(see export_model_colab.py) and place it in the models/ folder."
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2   # leave CPU free so the browser stays smooth during analysis
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = size or model_input_size(inp.shape, config.INPUT_SIZE)
        self.name = f"YOLOv8n {self.size}px (ONNX, CPU)"

    def detect(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        scale = self.size / max(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_x, pad_y = (self.size - nw) / 2, (self.size - nh) / 2
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[int(pad_y):int(pad_y) + nh, int(pad_x):int(pad_x) + nw] = resized

        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})[0]
        return decode_yolov8(output, scale, int(pad_x), int(pad_y),
                             config.CONF_THRESHOLD, config.NMS_THRESHOLD,
                             config.CLASSES, w, h)


class MotionDetector:
    name = "Motion blobs (fallback, not a classifier)"

    def __init__(self, min_area: int = 600):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=32,
                                                      detectShadows=False)
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> list[dict]:
        mask = self.bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            out.append({"box": [float(x), float(y), float(x + w), float(y + h)],
                        "conf": 0.5, "label": "motion"})
        return out


def load_detector(prefer_yolo: bool = True):
    """Return (detector, warning_or_None). Falls back to motion blobs if no model."""
    if prefer_yolo:
        try:
            return YoloOnnxDetector(), None
        except Exception as e:  # missing model, onnxruntime not installed, bad file
            return MotionDetector(), f"YOLO unavailable ({e}). Using motion fallback."
    return MotionDetector(), None
