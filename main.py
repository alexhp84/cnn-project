"""
FastAPI backend for the CNN Project.

Keep this file as main.py. Put the website in index.html and model artefacts
(.keras/.h5) in the project directory or a subdirectory named models/.
Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
from pathlib import Path
from typing import Any
import logging

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cnn-project")

app = FastAPI(
    title="CNN Image Analysis API",
    description=(
        "API for the CNN Image Analysis project. Classifies uploaded images using "
        "available emotion, fruit, and face-mask models; exposes health status and "
        "serves the web interface and model-result artefacts. Use POST /predict "
        "to submit an image. Interactive documentation is available at /docs."
    ),
    version="1.1.0",
    contact={
        "name": "CNN Project API",
    },
    openapi_tags=[
        {"name": "Website", "description": "HTML pages for the image-analysis application and results dashboard."},
        {"name": "Prediction", "description": "Image upload and classification operations."},
        {"name": "System", "description": "Backend health and model-loading status."},
        {"name": "Model results", "description": "Training graphs and history artefacts saved in the results directory."},
    ],
)

# Keep the labels in the same order used when each model was trained.
LABELS = {
    "fer": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
    "emotion": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
    "fruit": ["Apple", "Avocado", "Banana", "Eggplant", "Tomato"],
    "fruits": ["Apple", "Avocado", "Banana", "Eggplant", "Tomato"],
    "mask": ["with_mask", "without_mask"],
}

MODEL_KEYWORDS = {
    "fer": ("fer", "emotion", "angry"),
    "fruits": ("fruit", "fruits"),
    "mask": ("mask", "face_mask", "facemask"),
}
SUPPORTED_MODEL_SUFFIXES = {".keras", ".h5", ".hdf5"}
_loaded_models: dict[str, Any] = {}
_model_errors: dict[str, str] = {}


def find_model_file(key: str) -> Path | None:
    """Find a model file by keyword; explicit models/ folder is preferred."""
    candidates = []
    for root in (MODELS_DIR, BASE_DIR):
        if root.exists():
            candidates.extend(
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_MODEL_SUFFIXES
            )
    # Don't accidentally search through virtual environments or hidden folders.
    candidates = [
        p for p in candidates
        if not any(part.startswith(".") or part in {"venv", ".venv", "__pycache__"}
                   for part in p.relative_to(BASE_DIR).parts)
    ]
    keywords = MODEL_KEYWORDS[key]
    matches = [
        p for p in candidates
        if any(word in p.stem.lower() for word in keywords)
    ]
    # Avoid duplicate paths if models/ is also nested below the project.
    matches = sorted(set(matches), key=lambda p: (0 if MODELS_DIR in p.parents else 1, len(p.parts)))
    return matches[0] if matches else None


def load_models() -> None:
    """Load any model files that can be matched to the supported model types."""
    try:
        from importlib import import_module

        load_model = import_module("tensorflow.keras.models").load_model
    except Exception as exc:
        logger.exception("TensorFlow could not be imported")
        for key in MODEL_KEYWORDS:
            _model_errors[key] = f"TensorFlow is unavailable: {exc}"
        return

    for key in MODEL_KEYWORDS:
        path = find_model_file(key)
        if path is None:
            _model_errors[key] = (
                f"No {key} model file found. Place its .keras/.h5 file in "
                f"'{MODELS_DIR.name}/' and include '{key}' in its filename."
            )
            logger.warning(_model_errors[key])
            continue
        try:
            _loaded_models[key] = load_model(path, compile=False)
            logger.info("Loaded %s model from %s", key, path)
        except Exception as exc:
            _model_errors[key] = f"Could not load {path.name}: {exc}"
            logger.exception("Failed loading %s model", key)


@app.on_event("startup")
def startup_event() -> None:
    load_models()


def label_names_for(key: str, output_count: int) -> list[str]:
    labels = LABELS[key]
    # Use configured labels only when their count matches the model output.
    if len(labels) == output_count:
        return labels
    return [f"class_{i}" for i in range(output_count)]


def prepare_image(image: Image.Image, model: Any) -> np.ndarray:
    """Resize and colour-convert using the model's declared input shape."""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    # Expected NHWC or NCHW input; standard Keras image models are NHWC.
    height = int(shape[1]) if len(shape) == 4 and shape[1] not in (None, 1, 3, 4) else 224
    width = int(shape[2]) if len(shape) == 4 and shape[2] not in (None, 1, 3, 4) else 224
    channels = shape[-1] if len(shape) == 4 else 3

    if channels == 1:
        image = image.convert("L").resize((width, height))
        array = np.asarray(image, dtype=np.float32)[..., np.newaxis]
    else:
        image = image.convert("RGB").resize((width, height))
        array = np.asarray(image, dtype=np.float32)

    # Common image models are trained on 0..1 pixel values. If your training
    # pipeline used a different preprocessing scheme, match it here.
    array = array / 255.0
    return np.expand_dims(array, axis=0)


def prediction_for(key: str, image: Image.Image) -> dict[str, Any]:
    model = _loaded_models[key]
    batch = prepare_image(image, model)
    raw = np.asarray(model.predict(batch, verbose=0))
    scores = raw[0].astype(float).reshape(-1)

    # Most classification models return softmax probabilities. If outputs do
    # not look like probabilities, convert logits to a stable softmax.
    if np.any(scores < 0) or not np.isclose(float(scores.sum()), 1.0, atol=0.05):
        scores = scores - np.max(scores)
        scores = np.exp(scores)
        scores = scores / max(float(scores.sum()), 1e-12)

    labels = label_names_for(key, len(scores))
    probabilities = {label: float(score) for label, score in zip(labels, scores)}
    best_index = int(np.argmax(scores))
    return {
        "model": {"fer": "Emotion", "fruits": "Fruit", "mask": "Mask"}[key],
        "prediction": labels[best_index],
        "confidence": float(scores[best_index]),
        "probabilities": probabilities,
    }


@app.get(
    "/",
    tags=["Website"],
    summary="Open the image-analysis application",
    description="Returns the index.html frontend. This route serves HTML rather than JSON.",
    response_description="The image-analysis web page.",
    responses={404: {"description": "index.html was not found in the project directory."}},
)
def home():
    index_file = BASE_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html was not found in the project directory.")
    return FileResponse(index_file)


@app.get(
    "/results.html",
    tags=["Website"],
    summary="Open the model results dashboard",
    description="Returns results.html, which displays model metrics and the pre-generated training graphs.",
    response_description="The model results HTML page.",
    responses={404: {"description": "results.html was not found in the project directory."}},
)
def results_page():
    results_file = BASE_DIR / "results.html"
    if not results_file.exists():
        raise HTTPException(status_code=404, detail="results.html was not found in the project directory.")
    return FileResponse(results_file)


@app.get(
    "/health",
    tags=["System"],
    summary="Check API and model status",
    description=(
        "Returns API status, the keys of models successfully loaded, and any model "
        "loading or prediction errors. A status of 'ok' means the API process is responding; "
        "check loaded_models to confirm which classifiers are available."
    ),
    response_description="Current backend and model status.",
)
def health():
    return {
        "status": "ok",
        "loaded_models": sorted(_loaded_models.keys()),
        "model_errors": _model_errors,
    }


@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Classify an uploaded image",
    description=(
        "Upload one image using multipart/form-data with the field name 'file'. "
        "The API runs each available emotion, fruit, and mask model and returns "
        "the top prediction, confidence, and per-class scores for each available model. "
        "Supported images depend on Pillow's installed codecs; PNG and JPEG are recommended. "
        "Images are resized to each model's input dimensions and pixel values are normalised to 0–1."
    ),
    response_description="A list of predictions produced by the available models.",
    responses={
        400: {"description": "No image supplied, empty upload, unsupported content type, or invalid/unsupported image."},
        503: {"description": "No classification models are available. The response includes model_errors."},
    },
)
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
        raise HTTPException(
            status_code=400,
            detail="This image format is unsupported or the file is invalid. Try PNG or JPEG.",
        ) from exc

    results = []
    for key in ("fer", "fruits", "mask"):
        if key not in _loaded_models:
            continue
        try:
            results.append(prediction_for(key, image))
        except Exception as exc:
            logger.exception("Prediction failed for model %s", key)
            _model_errors[key] = str(exc)

    if not results:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "No classification models are currently available.",
                "model_errors": _model_errors,
            },
        )

    return {"filename": file.filename, "results": results}

@app.get(
    "/results/{filename}",
    tags=["Model results"],
    summary="Retrieve a model-result artefact",
    description=(
        "Serves a file from the project's results/ directory, such as a confusion-matrix PNG, "
        "training-curve PNG, or model-history JSON. Supply only the filename, for example "
        "'fer_curves.png' or 'fer_history.json'. Files must already exist in results/."
    ),
    response_description="The requested PNG image or JSON history file.",
    responses={
        404: {"description": "The requested result file does not exist."},
    },
)
def get_result_file(filename: str):
    # Prevent path traversal and only serve files directly inside results/.
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise HTTPException(status_code=404, detail="Result file not found.")
    result_file = RESULTS_DIR / filename
    if not result_file.is_file():
        raise HTTPException(status_code=404, detail="Result file not found.")
    return FileResponse(result_file, filename=result_file.name)