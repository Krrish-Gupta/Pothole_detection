from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
WEIGHTS_PATH  = BASE_DIR / "weights" / "best.pt"
DB_PATH       = BASE_DIR / "detections.db"
TEMP_DIR      = BASE_DIR / "temp_images"
TEMP_DIR.mkdir(exist_ok=True)

# ── Model settings ─────────────────────────────────────────────────────────────
IMG_SIZE       = 640          # YOLO input resolution
CONF_THRESHOLD = 0.50         # minimum confidence to accept a detection
IOU_THRESHOLD  = 0.45         # NMS overlap threshold

# NOTE: When adding crack/anomaly detection via RDD2022, update this list
# to match your retrained model's classes, e.g.:
# CLASS_NAMES = ["pothole", "crack", "alligator_crack", "patch"]
CLASS_NAMES    = ["pothole"]  # must match your data.yaml

# ── Severity thresholds (used only if falling back to rule-based logic) ────────
SEVERITY_LOW_MAX    = 0.02    # < 2%  of image area → Low
SEVERITY_MEDIUM_MAX = 0.08    # 2–8%  of image area → Medium
                              # > 8%  of image area → High

# ── Dashboard ──────────────────────────────────────────────────────────────────
APP_TITLE      = "Urban Road Defect Detection"
MAX_UPLOAD_MB  = 10
