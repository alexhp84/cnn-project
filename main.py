"""CNN image-analysis FastAPI backend.

Expected layout:
  main.py
  index.html
  static/styles.css
  static/app.js
  models/fer_model.keras
  models/fer_class_names.json
  models/fruits_model.keras
  models/fruits_class_names.json
  models/mask_model.keras
  models/mask_class_names.json

The user selects one specialist classifier: Emotion, Fruit, or Mask. No router model is used.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
SUPPORTED_SUFFIXES = {".keras", ".h5", ".hdf5"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cnn-project")

CONFIDENCE_THRESHOLD = 0.60

API_DESCRIPTION = """
Classifies uploaded images using the specialist CNN selected by the user: **Emotion**, **Fruit**, or **Mask**. The selected classifier is run directly; no routing model is required.

### Prediction
Use `POST /predict` with an image in the `file` field and a `classifier` field set to `emotion`, `fruit`, or `mask`. Predictions below the configured confidence threshold (60%) are marked `UNABLE TO IDENTIFY`.

### Models
Place the three specialist `.keras` files and their class-label JSON files in `models/`. `GET /health` reports which models loaded and any loading errors.

### Limitations
This is an educational image-classification project, not a medical or security decision system.
Confidence scores are model outputs, not guarantees of correctness. Authentication and upload-size
limits are not configured; add appropriate controls before exposing the service publicly.
"""

app = FastAPI(
    title="CNN Image Analysis API",
    summary="Choose Emotion, Fruit, or Mask and classify an image with the corresponding CNN model.",
    description=API_DESCRIPTION,
    version="2.2.0",
    contact={"name": "CNN Project API"},
    license_info={"name": "Educational project"},
    openapi_tags=[
        {
            "name": "Prediction",
            "description": "Upload an image and receive the prediction from the specialist model selected by the router."
        },
        {
            "name": "System",
            "description": "Inspect API and model-loading status."
        },
        {
            "name": "Model results",
            "description": "Download a generated result file by its filename."
        }
    ],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MODEL_FILES = {
    "emotion": ("fer_model.keras", "fer_class_names.json"),
    "fruit": ("fruits_model.keras", "fruits_class_names.json"),
    "mask": ("mask_model.keras", "mask_class_names.json"),
}
DISPLAY_NAMES = {"emotion": "Emotion", "fruit": "Fruit", "mask": "Mask"}
_models: dict[str, Any] = {}
_labels: dict[str, list[str]] = {}
_model_errors: dict[str, str] = {}


def _find_model(key: str) -> Path | None:
    expected = MODEL_FILES[key][0]
    direct = MODELS_DIR / expected
    if direct.is_file():
        return direct
    stem = Path(expected).stem.lower()
    for root in (MODELS_DIR, BASE_DIR):
        if root.exists():
            for path in root.rglob("*"):
                if (path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
                        and stem in path.stem.lower()
                        and not any(p.startswith(".") or p in {"venv", ".venv", "__pycache__"}
                                    for p in path.relative_to(BASE_DIR).parts)):
                    return path
    return None


def _load_labels(key: str, output_count: int) -> list[str]:
    labels_file = MODELS_DIR / MODEL_FILES[key][1]
    if labels_file.is_file():
        try:
            data = json.loads(labels_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                labels = [str(x) for x in data]
            elif isinstance(data, dict):
                if all(str(k).isdigit() for k in data):
                    labels = [str(data[k]) for k in sorted(data, key=lambda x: int(x))]
                elif all(str(v).isdigit() for v in data.values()):
                    labels = [str(k) for k, v in sorted(data.items(), key=lambda item: int(item[1]))]
                else:
                    labels = [str(v) for v in data.values()]
            else:
                labels = []
            if len(labels) == output_count:
                return labels
            logger.warning("%s has %d labels but model outputs %d classes", labels_file, len(labels), output_count)
        except Exception:
            logger.exception("Could not read labels file %s", labels_file)
    fallback = {
        "emotion": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
        "fruit": ["Apple", "Avocado", "Banana", "Eggplant", "Tomato"],
        "mask": ["with_mask", "without_mask"],
    }
    labels = fallback.get(key, [])
    if len(labels) == output_count:
        return labels
    return [f"class_{i}" for i in range(output_count)]


def _load_models() -> None:
    try:
        from tensorflow.keras.models import load_model
    except Exception as exc:
        for key in MODEL_FILES:
            _model_errors[key] = f"TensorFlow import failed: {exc}"
        logger.exception("TensorFlow could not be imported")
        return

    for key in MODEL_FILES:
        path = _find_model(key)
        if path is None:
            _model_errors[key] = f"Model file not found: {MODEL_FILES[key][0]}"
            logger.warning(_model_errors[key])
            continue
        try:
            _models[key] = load_model(path, compile=False)
            output_shape = _models[key].output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]
            _labels[key] = _load_labels(key, int(output_shape[-1]))
            logger.info("Loaded %s from %s with labels %s", key, path, _labels[key])
            _model_errors.pop(key, None)
        except Exception as exc:
            _model_errors[key] = str(exc)
            logger.exception("Failed to load %s model", key)


@app.on_event("startup")
def startup() -> None:
    _load_models()


def _prepare_image(image: Image.Image, model: Any) -> np.ndarray:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if len(shape) == 4:
        height = int(shape[1]) if shape[1] is not None else 224
        width = int(shape[2]) if shape[2] is not None else 224
        channels = shape[-1] if shape[-1] in (1, 3, 4) else 3
    else:
        height, width, channels = 224, 224, 3

    if channels == 1:
        image = image.convert("L").resize((width, height))
        arr = np.asarray(image, dtype=np.float32)[..., None]
    else:
        image = image.convert("RGB").resize((width, height))
        arr = np.asarray(image, dtype=np.float32)
    # Match this to the preprocessing used when training each model.
    arr /= 255.0
    return np.expand_dims(arr, axis=0)


def _predict_scores(key: str, image: Image.Image) -> tuple[list[str], np.ndarray]:
    model = _models[key]
    raw = np.asarray(model.predict(_prepare_image(image, model), verbose=0))
    scores = raw[0].astype(float).reshape(-1)
    if np.any(scores < 0) or not np.isclose(float(scores.sum()), 1.0, atol=0.05):
        scores = scores - np.max(scores)
        scores = np.exp(scores)
        scores = scores / max(float(scores.sum()), 1e-12)
    labels = _labels.get(key) or [f"class_{i}" for i in range(len(scores))]
    if len(labels) != len(scores):
        labels = [f"class_{i}" for i in range(len(scores))]
    return labels, scores


def _normalise_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def _make_result(key: str, labels: list[str], scores: np.ndarray) -> dict[str, Any]:
    idx = int(np.argmax(scores))
    confidence = float(scores[idx])
    probabilities = {label: float(score) for label, score in zip(labels, scores)}

    # Reject low-confidence predictions on the server, before the UI displays them.
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "model": DISPLAY_NAMES[key],
            "prediction": "UNABLE TO IDENTIFY",
            "confidence": confidence,
            "threshold": CONFIDENCE_THRESHOLD,
            "status": "LOW_CONFIDENCE",
            "message": f"Prediction did not meet the {CONFIDENCE_THRESHOLD:.0%} confidence threshold."
        }

    return {
        "model": DISPLAY_NAMES[key],
        "prediction": labels[idx],
        "confidence": confidence,
        "threshold": CONFIDENCE_THRESHOLD,
        "status": "IDENTIFIED",
        "message": f"Prediction met the {CONFIDENCE_THRESHOLD:.0%} confidence threshold.",
        "probabilities": probabilities,
    }

@app.get(
    "/",
    tags=["Web pages"],
    summary="Serve the project home page",
    description="Returns the main HTML page for the CNN image-classification project.",
)
def home():
    path = BASE_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(path)


@app.get(
    "/results.html",
    tags=["Web pages"],
    summary="Serve the results page",
    description="Returns the HTML page used to display classification results.",
)
def results_page():
    path = BASE_DIR / "results.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="results.html not found")
    return FileResponse(path)


@app.get(
    "/health",
    tags=["System"],
    summary="Check API and model status",
    description="Returns API status, the names of successfully loaded models, and any model-loading errors.",
    response_description="Current API and model-loading status.",
)
def health():
    return {"status": "ok", "loaded_models": sorted(_models), "model_errors": _model_errors}


@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Classify an uploaded image",
    description=(
        "Upload an image as multipart form data using the `file` field and select `classifier` "
        "as `emotion`, `fruit`, or `mask`. The selected specialist model returns the prediction. "
        "PNG and JPEG images are recommended."
    ),
    response_description="The router class and the selected specialist model's prediction results.",
    responses={
        400: {"description": "The upload is empty, is not identified as an image, or cannot be decoded."},
        422: {"description": "The classifier field is missing or invalid."},
        503: {"description": "The selected specialist model is unavailable."},
    },
)
async def predict(file: UploadFile = File(...), classifier: str = Form(..., description="Select one: emotion, fruit, or mask", pattern="^(emotion|fruit|mask)$")):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        from io import BytesIO
        with Image.open(BytesIO(contents)) as opened:
            opened.load()
            image = opened.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Unsupported image. Try PNG or JPEG.") from exc

    # Run only the specialist explicitly selected in the UI.
    specialist = classifier.strip().lower()
    if specialist not in DISPLAY_NAMES:
        raise HTTPException(status_code=422, detail="classifier must be emotion, fruit, or mask.")
    if specialist not in _models:
        detail = _model_errors.get(specialist, f"The {specialist} model is unavailable.")
        raise HTTPException(status_code=503, detail=detail)
    try:
        labels, scores = _predict_scores(specialist, image)
        return {
            "filename": file.filename,
            "classifier": specialist,
            "results": [_make_result(specialist, labels, scores)],
        }
    except Exception as exc:
        logger.exception("%s prediction failed", specialist)
        raise HTTPException(status_code=500, detail=f"{DISPLAY_NAMES[specialist]} prediction failed.") from exc


@app.get(
    "/results/{filename}",
    tags=["Model results"],
    summary="Retrieve a result file",
    description="Returns a file from the project's `results/` directory by filename.",
    response_description="The requested result file.",
    responses={404: {"description": "No result file exists with the supplied filename."}},
)
def get_result_file(filename: str):
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise HTTPException(status_code=404, detail="Result file not found.")
    path = BASE_DIR / "results" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found.")
    return FileResponse(path, filename=path.name)
