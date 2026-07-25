"""
REST API for the pothole detection pipeline.

This file does NOT touch your existing detection logic — it just exposes
your existing pipeline.py functions over HTTP so other systems (or a
JavaScript frontend) can call them.

Run with:  python api.py
Then test with:  curl -X POST -F "image=@test.jpg" http://localhost:5000/api/detect
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path

from pipeline import run_pipeline
from database import get_all_results, get_severity_summary

app = Flask(__name__)
CORS(app)  # allows index.html (opened as a local file) to call this API


@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Accepts an image file upload, runs the detection pipeline,
    and returns the result as JSON.

    Request:  multipart/form-data with a field named "image"
    Response: JSON with defect_count, severity, confidence, bboxes, etc.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field 'image'."}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        result = run_pipeline(image_file, filename=image_file.filename)
    except Exception as e:
        # Never let the server crash silently — always return a JSON error
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

    # bboxes/annotated_path are not JSON-serialisable as-is (Path object),
    # so we convert before sending back
    return jsonify({
        "defect_count": result["defect_count"],
        "severity": result["severity"],
        "confidence": round(result["confidence"], 4),
        "bboxes": result["bboxes"],
        "inference_ms": result["inference_ms"],
        "db_id": result["db_id"],
        "annotated_image_url": f"/api/image/{Path(result['annotated_path']).name}"
    })


@app.route("/api/image/<filename>", methods=["GET"])
def get_annotated_image(filename):
    """Serves the annotated image so the frontend can display it."""
    from config import TEMP_DIR
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "Image not found."}), 404
    return send_file(str(file_path), mimetype="image/jpeg")


@app.route("/api/results", methods=["GET"])
def results():
    """Returns detection history as JSON (mirrors the Streamlit History page)."""
    df = get_all_results()
    if df.empty:
        return jsonify([])

    # Drop bbox_data (too heavy) and convert timestamps to strings for JSON
    df = df.drop(columns=["bbox_data"], errors="ignore")
    df["timestamp"] = df["timestamp"].astype(str)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/summary", methods=["GET"])
def summary():
    """Returns severity counts as JSON (mirrors the Dashboard KPI cards)."""
    return jsonify(get_severity_summary())


@app.route("/api/health", methods=["GET"])
def health():
    """Simple check to confirm the API is running."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
