"""All tunable settings in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "yolov8n.onnx"
MODEL_PATH_640 = ROOT / "models" / "yolov8n-640.onnx"   # optional, more accurate (used automatically if present)
VIDEO_DIR = ROOT / "data" / "videos"
CACHE_DIR = ROOT / "data" / "cache"

# Detection
INPUT_SIZE = 320            # must match the imgsz used at export time
CONF_THRESHOLD = 0.40       # people
BAG_CONF_THRESHOLD = 0.20   # bags are small and often blurred, so we accept weaker detections for them
NMS_THRESHOLD = 0.45
# COCO class ids we care about -> label
CLASSES = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}

# Video
PROC_WIDTH = 640            # frames are resized to this width before processing
FRAME_STRIDE = 2            # process every Nth frame (raise if laptop is slow)

# Tracking
TRACK_IOU_MIN = 0.2
TRACK_MAX_AGE = 15          # processed frames a track may go unseen before it is closed
TRACK_MIN_HITS = 4          # ignore tracks seen fewer times than this (noise)
TRACK_MIN_HITS_BAG = 2      # a thrown or carried bag may be visible for only a few frames
BAG_SOLO_CONF = 0.45        # ...and a bag seen in just ONE frame is still kept if the model is this sure
MIN_PERSON_H = 40           # ignore "person" detections shorter than this many pixels (distant blobs, reflections)

# ---- Behaviour analysis (Step 2) ----
PERSON_LABELS = {"person", "motion"}          # "motion" = fallback detector blobs
BAG_LABELS = {"backpack", "handbag", "suitcase"}
LOITER_S = 8.0           # seconds inside the restricted zone before it counts as loitering
ABANDON_S = 8.0          # seconds an object stays still with nobody near it
STILL_SPEED = 0.25       # body-heights per second below which a person counts as standing still
RUN_SPEED = 1.8          # body-heights per second (90th percentile) above which movement counts as rapid
MIN_SEVERITY = 0         # hide incidents scoring below this (0 = show every rule-triggered incident)
ANOMALY_MIN_TRACKS = 8   # Isolation Forest needs at least this many person tracks to be meaningful
# Severity weights (points). Each factor contributes weight * value(0..1); total is capped at 100.
WEIGHTS = {"zone_entry": 20, "dwell": 40, "abandoned": 60, "running": 30, "anomaly": 20}
MIN_TRACK_S = 1.0        # tracks that exist for less than this many seconds are treated as detector noise and ignored
