import numpy as np
from pathlib import Path
from ultralytics import YOLO
from config import WEIGHTS_PATH, CONF_THRESHOLD, IOU_THRESHOLD

# Load model once at module level — avoids reloading on every inference call
_model = None

def _get_model() -> YOLO:
    global _model
    if _model is None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"Model weights not found at {WEIGHTS_PATH}.\n"
                "Run train.py first, then copy best.pt into weights/."
            )
        _model = YOLO(str(WEIGHTS_PATH))
    return _model


def run_inference(img_bgr: np.ndarray, conf_override: float = None) -> dict:
    """
    Run YOLOv8 inference on a BGR numpy array.

    Args:
        img_bgr      : image as a BGR numpy array
        conf_override: if given, overrides config.CONF_THRESHOLD for this call
                        (used by the dashboard's confidence slider)

    Returns a dict with:
        bboxes      : list of [x1, y1, x2, y2] pixel coords
        confidences : list of float
        class_ids   : list of int
        img_h, img_w: original image dimensions
    """
    model = _get_model()
    h, w  = img_bgr.shape[:2]
    conf  = conf_override if conf_override is not None else CONF_THRESHOLD

    results = model.predict(
        source    = img_bgr,
        conf      = conf,
        iou       = IOU_THRESHOLD,
        imgsz     = 640,
        verbose   = False,
    )

    r          = results[0]
    bboxes     = r.boxes.xyxy.cpu().numpy().tolist()
    confidences= r.boxes.conf.cpu().numpy().tolist()
    class_ids  = r.boxes.cls.cpu().numpy().tolist()

    return {
        "bboxes":      bboxes,
        "confidences": confidences,
        "class_ids":   class_ids,
        "img_h":       h,
        "img_w":       w,
    }