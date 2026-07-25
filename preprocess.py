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
        # Streamlit UploadedFile / Flask FileStorage → bytes
        arr = np.frombuffer(source.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Could not read image from {source}")
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Resize to model input size
    2. Denoise  (handles compressed JPEG artifacts from phone cameras)
    3. Normalise pixel values to 0–1 float32

    Note: YOLOv8 re-normalises internally before inference.
    The float32 output here is used for the annotated preview image.
    """
    # 1. Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # 2. Denoise — parameters tuned for road surface textures
    #    h=10 (luminance), hColor=10, templateWindowSize=7, searchWindowSize=21
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # 3. Normalise
    img = img.astype(np.float32) / 255.0

    return img


def save_annotated(
    original_bgr: np.ndarray,
    bboxes: list[list[float]],
    confidences: list[float],
    severity: str,
    filename: str
) -> Path:
    """
    Draw bounding boxes + severity label on the original image and save to
    the temp directory. Returns the saved file path.

    bboxes: list of [x1, y1, x2, y2] in PIXEL coordinates (xyxy format)
    """
    img = original_bgr.copy()
    h, w = img.shape[:2]

    SEVERITY_COLOURS = {
        "Low":    (57, 153, 34),    # green  (BGR)
        "Medium": (23, 117, 186),   # amber
        "High":   (74, 75, 226),    # red
        "None":   (128, 128, 128),  # gray
        "Ignore": (128, 128, 128),  # gray
    }
    colour = SEVERITY_COLOURS.get(severity, (128, 128, 128))

    for bbox, conf in zip(bboxes, confidences):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness=2)
        label = f"pothole {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Severity badge in top-left corner
    badge = f"Severity: {severity}"
    cv2.rectangle(img, (10, 10), (200, 40), colour, -1)
    cv2.putText(img, badge, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    out_path = TEMP_DIR / filename
    cv2.imwrite(str(out_path), img)
    return out_path
