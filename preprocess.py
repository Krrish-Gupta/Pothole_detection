import cv2
import numpy as np
from pathlib import Path
from config import IMG_SIZE, TEMP_DIR


def load_image(source) -> np.ndarray:
    """
    Accept a file path (str/Path) or raw bytes from Streamlit uploader
    OR a Flask FileStorage object (both expose .read()).
    Returns a BGR numpy array.
    """
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
    else:
        arr = np.frombuffer(source.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Could not read image from {source}")
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    """Resize, denoise, and normalise — used to produce the display preview."""
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img = img.astype(np.float32) / 255.0
    return img


def save_annotated(
    original_bgr: np.ndarray,
    bboxes: list[list[float]],
    confidences: list[float],
    severity: str,
    filename: str,
    draw_labels: bool = True,
    per_box_severity: list[str] = None,
) -> Path:
    """
    Draw bounding boxes + severity label on the original image and save to
    the temp directory. Returns the saved file path.

    Args:
        draw_labels       : if False, boxes are still drawn but confidence
                             text labels are omitted (cleaner image for quick scanning)
        per_box_severity  : optional list matching bboxes — colours each box by
                             its own severity instead of the whole image's worst severity
    """
    img = original_bgr.copy()

    SEVERITY_COLOURS = {
        "Low":    (57, 153, 34),
        "Medium": (23, 117, 186),
        "High":   (74, 75, 226),
        "None":   (128, 128, 128),
        "Ignore": (128, 128, 128),
    }
    default_colour = SEVERITY_COLOURS.get(severity, (128, 128, 128))

    for i, (bbox, conf) in enumerate(zip(bboxes, confidences)):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        box_sev = per_box_severity[i] if per_box_severity else severity
        colour = SEVERITY_COLOURS.get(box_sev, default_colour)

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness=2)

        if draw_labels:
            label = f"pothole {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    badge = f"Severity: {severity}"
    cv2.rectangle(img, (10, 10), (200, 40), default_colour, -1)
    cv2.putText(img, badge, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    out_path = TEMP_DIR / filename
    cv2.imwrite(str(out_path), img)
    return out_path