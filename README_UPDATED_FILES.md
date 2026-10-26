# Updated CNN Project files

## Contents
- `main.py`: FastAPI backend; `/predict` uses the router model to select exactly one specialist classifier.
- `index.html`: upload/camera interface and restored navigation buttons.
- `static/styles.css`: styling.
- `static/app.js`: upload, camera, API call and single-result rendering.

## Required model files
Put these model files under `models/`:
- `router_model.keras` — required to choose a single classifier
- `fer_model.keras`
- `fruits_model.keras`
- `mask_model.keras`

The code also reads class-name JSON files if available:
- `router_class_names.json`
- `fer_class_names.json`
- `fruits_class_names.json`
- `mask_class_names.json`

**Important:** the router's output labels must clearly identify the image type, such as `face`, `fruit`, or `mask`. If your router has different labels, update `_route_to_specialist()` in `main.py` to match the exact labels. If you do not have a router model, this version intentionally returns an error rather than displaying three unrelated predictions.

## Run
Install dependencies if needed, then run from the directory containing `main.py`:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Open `http://127.0.0.1:8000/`.
Navigation buttons: `/docs` (API docs), `/health` (model status), `/results.html` (results dashboard).

This archive does not replace your existing trained models or `results.html`. Keep those files in your project directory.
