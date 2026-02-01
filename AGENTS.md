# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This repository contains a small, production-oriented crop disease prediction system with two primary entrypoints:
- `app.py`: Streamlit-based UI for interactive predictions.
- `api.py`: FastAPI-based HTTP API for programmatic predictions.

Three model artifacts are included in the repo root and are treated as interchangeable backends:
- `crop_disease_model.pkl` — scikit-learn model (loaded via `joblib`).
- `crop_disease_model.keras` — TensorFlow/Keras SavedModel.
- `crop_disease_model.h5` — TensorFlow/Keras H5 model.

Both the UI and API share the same high-level behavior:
- On startup, they attempt to load a model from the current working directory, preferring `.pkl`, then `.keras`, then `.h5`.
- If no model is found, the app/API will not function and should fail fast.

## Architecture and Code Structure

### Streamlit UI (`app.py`)

- Uses `st.cache_resource`-decorated `load_model()` to lazily load and cache the model from the repo root.
- After loading, it stores `(model, model_type)` in module scope and exposes a simple numeric feature input form:
  - `num_features` determines how many numeric inputs to render.
  - Inputs are collected into a Python list and converted to a `numpy` array of shape `(1, -1)`.
- Prediction flow:
  - For `model_type == "pkl"`: calls `model.predict(X)` and displays the raw prediction.
  - For Keras/TensorFlow models: calls `model.predict(X)` and then applies `np.argmax(..., axis=1)` to get a class index.
- All logic lives in a single file; there is *no* separate service, domain, or data-access layer.
- The code assumes models accept a flat numeric feature vector and does not perform validation beyond numeric conversion.

### FastAPI backend (`api.py`)

- Creates a `FastAPI` app instance at module import time (`app = FastAPI(...)`).
- Defines an internal `load_model()` helper (duplicated logic from `app.py`) that loads one of the model files in the same priority order.
- Loads `(model, model_type)` at module import; this will run once per process in typical ASGI deployments.
- Uses a simple Pydantic schema:
  - `PredictionRequest` with a single field `features: list`.
  - The API expects `features` to be a flat list of numeric values.
- Endpoints:
  - `GET /` — health/info endpoint, returns a simple JSON status message.
  - `POST /predict` — performs inference:
    - Converts `features` into a `(1, -1)` numpy array.
    - For `pkl` models: `model.predict(X)` and returns `.tolist()`.
    - For Keras/TensorFlow models: `model.predict(X)`, `np.argmax(..., axis=1).tolist()`.
- The API does not currently do input validation beyond structural shape and will raise runtime errors if types are incompatible with the underlying model.

### Shared assumptions and constraints

- All model files are expected to live in the repository root and be readable from the current working directory.
- Model loading is performed eagerly in `api.py` and lazily (but cached) in `app.py`.
- There is no central utilities module; any cross-cutting changes to model loading or prediction behavior require edits in both `app.py` and `api.py`.
- Dependencies are declared in `requirements.txt` and target CPU-friendly inference (`tensorflow-cpu`).

## Commands for Development

### Environment setup

Use Python with `pip` to install dependencies from `requirements.txt`:

- Install dependencies:
  - `pip install -r requirements.txt`

> Note: The large model files (`*.pkl`, `*.h5`, `*.keras`) are already checked in; do not attempt to open or diff them in tools that expect text.

### Running the Streamlit UI

- Start the Streamlit app from the repository root:
  - `streamlit run app.py`

This launches the browser-based UI where users can:
- Configure the number of numeric input features.
- Enter feature values.
- Trigger predictions with the loaded model.

### Running the FastAPI backend

The backend is designed to run with Uvicorn as the ASGI server.

- Start the API server with auto-reload in development:
  - `uvicorn api:app --reload --port 8000`

Key behaviors:
- On startup, the model is loaded once using the same logic as the Streamlit app.
- `GET /` can be used for health checks.
- `POST /predict` expects JSON of the form:

  ```json
  {
    "features": [1.0, 2.5, 3.3, 4.1]
  }
  ```

### Linting and tests

- There are currently **no dedicated linting or testing tools configured** in this repository (no linters or test frameworks are declared in `requirements.txt`).
- If you introduce linting or tests (e.g., `pytest`, `ruff`, `black`), update this section with the exact commands.

## Guidance for Future Changes

- When modifying model loading behavior (e.g., adding support for new file types, changing priority order), keep `app.py` and `api.py` in sync or refactor shared logic into a common helper module.
- Be mindful of backward compatibility with existing model files; any change in expected input shape or preprocessing should be coordinated with how models are trained.
- Avoid heavy computation during import beyond model loading; if startup time becomes an issue, consider lazy-loading strategies similar to `st.cache_resource` in non-Streamlit contexts.
