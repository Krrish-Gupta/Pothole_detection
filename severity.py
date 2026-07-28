"""
================================================================
severity.py — ML-Based Severity Classification using Random Forest
================================================================
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ── Training Data (Features + Labels) ────────────────────────
# Features: [ratio_pct, aspect_ratio, confidence]
# Labels: 0 = Low, 1 = Medium, 2 = High

X_train = np.array([
    # ratio_pct aspect_ratio confidence → Label
    [0.1, 1.0, 0.50], # Low
    [0.2, 1.1, 0.52], # Low
    [0.3, 0.9, 0.55], # Low
    [0.4, 1.2, 0.57], # Low
    [0.5, 1.0, 0.58], # Low
    [0.8, 1.3, 0.60], # Low
    [1.0, 1.1, 0.62], # Low

    [2.0, 1.5, 0.65], # Medium
    [3.0, 1.7, 0.67], # Medium
    [4.0, 1.8, 0.70], # Medium
    [5.0, 2.0, 0.72], # Medium
    [6.0, 1.9, 0.74], # Medium
    [7.0, 2.1, 0.75], # Medium

    [8.0, 2.3, 0.78], # High
    [9.0, 2.5, 0.80], # High
    [10.0, 2.7, 0.82], # High
    [12.0, 3.0, 0.85], # High
    [15.0, 3.2, 0.88], # High
    [20.0, 3.5, 0.90], # High
])

y_train = np.array([
    0, 0, 0, 0, 0, 0, 0, # Low
    1, 1, 1, 1, 1, 1, # Medium
    2, 2, 2, 2, 2, 2 # High
])

_LABELS = ["Low", "Medium", "High"]

# ── Train Random Forest ───────────────────────────────────────
_scaler = StandardScaler()
X_scaled = _scaler.fit_transform(X_train)

_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=5
)
_clf.fit(X_scaled, y_train)


# ── Main Classification Function ──────────────────────────────
def classify_severity(
    x1: int, y1: int, x2: int, y2: int,
    img_w: int, img_h: int,
    label: str = "pothole",
    confidence: float = 0.7
):
    """
    Classify pothole severity using Random Forest.

    Parameters
    ----------
    x1, y1, x2, y2 : int — Bounding box coordinates
    img_w, img_h : int — Image dimensions
    label : str — Detected class label from YOLO
    confidence : float — YOLO confidence score (0–1)

    Returns
    -------
    severity : str — "Low", "Medium", "High", or "Ignore"
    ratio_pct : float — Bbox area as % of image area
    area_px : int — Bbox area in pixels
    """

    # ── Filter non-pothole detections ────────────────────────
    if label.lower() != "pothole":
        return "Ignore", 0.0, 0

    # ── Guard: invalid box coordinates ───────────────────────
    if x2 <= x1 or y2 <= y1:
        return "Low", 0.0, 0

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    area_px = width * height
    img_area = max(img_w * img_h, 1)

    ratio_pct = round((area_px / img_area) * 100, 4)
    aspect_ratio = round(width / height, 4)

    # ── Random Forest Prediction ──────────────────────────────
    features = np.array([[ratio_pct, aspect_ratio, confidence]])
    features_scaled = _scaler.transform(features)
    pred = _clf.predict(features_scaled)[0]
    severity = _LABELS[pred]

    return severity, ratio_pct, area_px


# ── Color & Display Helpers ───────────────────────────────────

def get_severity_color_cv2(severity: str):
    """Returns BGR color tuple for OpenCV drawing."""
    return {
        "Low": (34, 197, 94),
        "Medium": (30, 150, 255),
        "High": (0, 0, 220),
        "Ignore": (200, 200, 200),
    }.get(severity, (200, 200, 200))


def get_severity_color_hex(severity: str):
    """Returns hex color string for Streamlit / HTML."""
    return {
        "Low": "#22c55e",
        "Medium": "#f97316",
        "High": "#dc2626",
        "Ignore": "#aaaaaa",
    }.get(severity, "#aaaaaa")


def get_severity_emoji(severity: str):
    """Returns a colored circle emoji for the severity level."""
    return {
        "Low": "🟢",
        "Medium": "🟠",
        "High": "🔴",
        "Ignore": "⚪",
    }.get(severity, "⚪")


def get_severity_score(ratio_pct: float) -> int:
    """Returns a 0–100 numeric severity score from area ratio."""
    return min(int(ratio_pct * 10), 100)


def get_feature_importance() -> dict:
    """
    Returns feature importance dict from the trained Random Forest.
    Useful for showing model insights in the Streamlit dashboard.
    """
    features = ["Area Ratio %", "Aspect Ratio", "Confidence"]
    importances = _clf.feature_importances_
    return dict(zip(features, importances))
