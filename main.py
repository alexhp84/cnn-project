"""CNN image-analysis FastAPI backend.

Expected layout:
  main.py
  index.html
  static/styles.css
  static/app.js
  models/router_model.keras
  models/router_class_names.json
  models/fer_model.keras
  models/fer_class_names.json
  models/fruits_model.keras
  models/fruits_class_names.json
  models/mask_model.keras
  models/mask_class_names.json

The router chooses ONE specialist classifier, so the UI does not show
Emotion, Fruit, and Mask results for every uploaded image.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
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

app = FastAPI(title="CNN Image Analysis API", version="2.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MODEL_FILES = {
    "router": ("router_model.keras", "router_class_names.json"),
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


def _route_to_specialist(router_label: str) -> str | None:
    """Map the router model's class label to one specialist model."""
    label = _normalise_label(router_label)
    if any(token in label for token in ("fruit", "apple", "banana", "avocado", "tomato", "eggplant")):
        return "fruit"
    if any(token in label for token in ("mask", "masked", "without_mask", "with_mask", "no_mask")):
        return "mask"
    if any(token in label for token in ("face", "person", "human", "emotion", "angry", "happy", "sad", "fear", "neutral", "surprise", "disgust")):
        return "emotion"
    # Handle common generic router labels such as "faces", "fruits", and "masks".
    if label in {"fruit", "fruits"}:
        return "fruit"
    if label in {"mask", "masks", "face_mask"}:
        return "mask"
    if label in {"face", "faces", "emotion", "emotions"}:
        return "emotion"
    return None


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
            "status": "THREAT TO ORDER. SENT TO ROOM 101, MINISTRY OF LOVE FOR EDUCATION"
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

@app.get("/", include_in_schema=False)
def home():
    path = BASE_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(path)


@app.get("/results.html", include_in_schema=False)
def results_page():
    path = BASE_DIR / "results.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="results.html not found")
    return FileResponse(path)


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "loaded_models": sorted(_models), "model_errors": _model_errors}


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
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

    # Preferred path: the router chooses exactly one specialist model.
    if "router" in _models:
        try:
            route_labels, route_scores = _predict_scores("router", image)
            route_label = route_labels[int(np.argmax(route_scores))]
            specialist = _route_to_specialist(route_label)
            logger.info("Router chose %r -> %r", route_label, specialist)
            if specialist and specialist in _models:
                labels, scores = _predict_scores(specialist, image)
                return {"filename": file.filename, "route": route_label,
                        "results": [_make_result(specialist, labels, scores)]}
            # Do not silently run all three models if routing is ambiguous.
            return {"filename": file.filename, "route": route_label, "results": [],
                    "message": f"Router returned '{route_label}', which could not be mapped to a specialist model."}
        except Exception as exc:
            logger.exception("Router prediction failed")
            raise HTTPException(status_code=500, detail=f"Router prediction failed: {exc}") from exc

    # Without a router, avoid presenting three unrelated predictions. Ask the
    # user to add router_model.keras, rather than pretending one is the right class.
    raise HTTPException(
        status_code=503,
        detail="The router model is required to select one classifier. Put router_model.keras in models/ and restart the server.",
    )


@app.get("/results/{filename}", tags=["Model results"])
def get_result_file(filename: str):
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise HTTPException(status_code=404, detail="Result file not found.")
    path = BASE_DIR / "results" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found.")
    return FileResponse(path, filename=path.name)
