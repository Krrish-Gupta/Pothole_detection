import time
from pathlib import Path
import numpy as np

from preprocess  import load_image, preprocess, save_annotated
from detect      import run_inference
from severity    import classify_severity
from database    import insert_result, init_db
from config      import TEMP_DIR

# Ensure DB and temp folder exist on first import
init_db()

# Rank used to pick the "worst" severity across multiple boxes in one image
_SEVERITY_RANK = {"Ignore": -1, "None": 0, "Low": 1, "Medium": 2, "High": 3}


def run_pipeline(source, filename: str | None = None) -> dict:
    """
    Full pipeline: load → preprocess → detect → classify → store → annotate.

    Args:
        source   : file path (str/Path) OR Streamlit UploadedFile / Flask FileStorage
        filename : display name saved to DB (auto-derived if not given)

    Returns a dict:
        annotated_path : Path to the annotated output image
        defect_count   : int
        severity       : "Low" | "Medium" | "High" | "None" | "Ignore"
        confidence     : float (max across all detections)
        bboxes         : list of [x1,y1,x2,y2]
        inference_ms   : float
        db_id          : int (inserted row ID)
    """
    # ── 1. Load ────────────────────────────────────────────────────────────────
    original_bgr = load_image(source)
    h, w         = original_bgr.shape[:2]
    fname        = filename or (Path(str(source)).name
                                if isinstance(source, (str, Path))
                                else "upload.jpg")

    # ── 2. Preprocess (for annotated preview — inference uses original) ────────
    _ = preprocess(original_bgr)   # kept for pipeline completeness / future use

    # ── 3. Detect ─────────────────────────────────────────────────────────────
    t0      = time.perf_counter()
    result  = run_inference(original_bgr)
    inf_ms  = (time.perf_counter() - t0) * 1000

    bboxes      = result["bboxes"]
    confidences = result["confidences"]

    # ── 4. Classify severity (Random Forest, per box) ───────────────────────────
    # The RF model classifies ONE box at a time, so we loop over every detected
    # box and keep the worst (highest-rank) severity as the image-level result.
    if bboxes:
        per_box_severities = []
        for (x1, y1, x2, y2), conf in zip(bboxes, confidences):
            sev, ratio_pct, area_px = classify_severity(
                x1, y1, x2, y2, w, h,
                label="pothole",   # update once multi-class detection is added
                confidence=conf
            )
            per_box_severities.append(sev)

        severity = max(per_box_severities, key=lambda s: _SEVERITY_RANK[s])
    else:
        severity = "None"

    max_conf = max(confidences, default=0.0)

    # ── 5. Save annotated image ────────────────────────────────────────────────
    out_fname    = f"annotated_{fname}"
    ann_path     = save_annotated(original_bgr, bboxes, confidences,
                                  severity, out_fname)

    # ── 6. Log to database ─────────────────────────────────────────────────────
    db_id = insert_result(
        image_name   = fname,
        image_path   = str(ann_path),
        defect_count = len(bboxes),
        severity     = severity,
        confidence   = max_conf,
        bboxes       = bboxes,
    )

    return {
        "annotated_path": ann_path,
        "defect_count":   len(bboxes),
        "severity":       severity,
        "confidence":     max_conf,
        "bboxes":         bboxes,
        "inference_ms":   round(inf_ms, 1),
        "db_id":          db_id,
    }