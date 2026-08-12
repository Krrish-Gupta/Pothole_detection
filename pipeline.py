import time
from pathlib import Path
import numpy as np

from preprocess  import load_image, preprocess, save_annotated
from detect      import run_inference
from severity    import classify_severity
from database    import insert_result, init_db
from config      import TEMP_DIR

init_db()

_SEVERITY_RANK = {"Ignore": -1, "None": 0, "Low": 1, "Medium": 2, "High": 3}


def run_pipeline(
    source,
    filename: str | None = None,
    conf_override: float = None,
    draw_labels: bool = True,
) -> dict:
    """
    Full pipeline: load → preprocess → detect → classify → store → annotate.

    Args:
        source        : file path (str/Path) OR file-like object (Streamlit/Flask upload)
        filename      : display name saved to DB (auto-derived if not given)
        conf_override : optional confidence threshold override (dashboard slider)
        draw_labels   : whether to draw confidence-score text labels on the annotated image

    Returns a dict with defect_count, severity (worst), low_count, medium_count,
    high_count, confidence, bboxes, per_box_severity, inference_ms, db_id, annotated_path.
    """
    original_bgr = load_image(source)
    h, w         = original_bgr.shape[:2]
    fname        = filename or (Path(str(source)).name
                                if isinstance(source, (str, Path))
                                else "upload.jpg")

    _ = preprocess(original_bgr)  # kept for pipeline completeness / future use

    t0      = time.perf_counter()
    result  = run_inference(original_bgr, conf_override=conf_override)
    inf_ms  = (time.perf_counter() - t0) * 1000

    bboxes      = result["bboxes"]
    confidences = result["confidences"]

    per_box_severity = []
    low_count = medium_count = high_count = 0

    for (x1, y1, x2, y2), conf in zip(bboxes, confidences):
        sev, ratio_pct, area_px = classify_severity(
            x1, y1, x2, y2, w, h, label="pothole", confidence=conf
        )
        per_box_severity.append(sev)
        if sev == "Low":
            low_count += 1
        elif sev == "Medium":
            medium_count += 1
        elif sev == "High":
            high_count += 1

    severity = (
        max(per_box_severity, key=lambda s: _SEVERITY_RANK[s])
        if per_box_severity else "None"
    )
    max_conf = max(confidences, default=0.0)

    out_fname = f"annotated_{fname}"
    ann_path = save_annotated(
        original_bgr, bboxes, confidences, severity, out_fname,
        draw_labels=draw_labels, per_box_severity=per_box_severity
    )

    db_id = insert_result(
        image_name   = fname,
        image_path   = str(ann_path),
        defect_count = len(bboxes),
        low_count    = low_count,
        medium_count = medium_count,
        high_count   = high_count,
        severity     = severity,
        confidence   = max_conf,
        bboxes       = bboxes,
    )

    return {
        "annotated_path":     ann_path,
        "defect_count":       len(bboxes),
        "severity":           severity,
        "low_count":          low_count,
        "medium_count":       medium_count,
        "high_count":         high_count,
        "confidence":         max_conf,
        "bboxes":             bboxes,
        "per_box_severity":   per_box_severity,
        "inference_ms":       round(inf_ms, 1),
        "db_id":              db_id,
    }