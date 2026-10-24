"""Small IoU-based multi-object tracker (no extra dependencies)."""
from __future__ import annotations

import numpy as np

import config


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


class Track:
    def __init__(self, tid: int, det: dict, frame_idx: int):
        self.id = tid
        self.label = det["label"]
        self.box = det["box"]
        self.conf = det["conf"]
        self.max_conf = det["conf"]
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.hits = 1
        self.missed = 0
        self.path = [self._foot(det["box"])]   # (x, y) of bottom-centre, per hit
        self.frames = [frame_idx]
        self.boxes = [list(det["box"])]

    @staticmethod
    def _foot(box):
        return [(box[0] + box[2]) / 2, box[3]]

    def update(self, det: dict, frame_idx: int):
        self.box = det["box"]
        self.conf = det["conf"]
        self.max_conf = max(self.max_conf, det["conf"])
        self.last_frame = frame_idx
        self.hits += 1
        self.missed = 0
        self.path.append(self._foot(det["box"]))
        self.frames.append(frame_idx)
        self.boxes.append(list(det["box"]))


def _confirmed(t) -> bool:
    if t.label in config.PERSON_LABELS:
        return t.hits >= config.TRACK_MIN_HITS
    # bags: a few frames are enough, and one very confident frame also counts
    return t.hits >= config.TRACK_MIN_HITS_BAG or t.max_conf >= config.BAG_SOLO_CONF


def _same_kind(a: str, b: str) -> bool:
    """backpack / handbag / suitcase flip between frames for the same bag, so treat them as one kind."""
    return a == b or (a in config.BAG_LABELS and b in config.BAG_LABELS)


class Tracker:
    def __init__(self):
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 1

    def step(self, detections: list[dict], frame_idx: int) -> list[Track]:
        """Match detections to tracks (greedy by IoU, same label). Returns confirmed tracks seen now."""
        pairs = []
        for ti, t in enumerate(self.active):
            for di, d in enumerate(detections):
                if not _same_kind(t.label, d["label"]):
                    continue
                score = iou(t.box, d["box"])
                if score >= config.TRACK_IOU_MIN:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            self.active[ti].update(detections[di], frame_idx)
            used_t.add(ti)
            used_d.add(di)

        for di, d in enumerate(detections):
            if di not in used_d:
                self.active.append(Track(self._next_id, d, frame_idx))
                self._next_id += 1

        still = []
        for ti, t in enumerate(self.active):
            if ti not in used_t and t.last_frame != frame_idx:
                t.missed += 1
            if t.missed > config.TRACK_MAX_AGE:
                self.finished.append(t)
            else:
                still.append(t)
        self.active = still

        return [t for t in self.active
                if t.last_frame == frame_idx and _confirmed(t)]

    def all_tracks(self) -> list[Track]:
        """Every confirmed track, closed or still open (call at end of video)."""
        return [t for t in self.finished + self.active if _confirmed(t)]
