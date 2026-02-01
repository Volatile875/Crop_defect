from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
import os

app = FastAPI(title="Crop Disease Prediction API")

# ------------------ MODEL LOADING ------------------

def load_model():
    if os.path.exists("crop_disease_model.pkl"):
        return joblib.load("crop_disease_model.pkl"), "pkl"
    elif os.path.exists("crop_disease_model.keras"):
        return tf.keras.models.load_model("crop_disease_model.keras"), "keras"
    elif os.path.exists("crop_disease_model.h5"):
        return tf.keras.models.load_model("crop_disease_model.h5"), "h5"
    else:
        return None, None

model, model_type = load_model()

# ------------------ REQUEST SCHEMA ------------------

class PredictionRequest(BaseModel):
    features: list

# ------------------ ROUTES ------------------

@app.get("/")
def home():
    return {"status": "Crop Disease Prediction API is running"}

@app.post("/predict")
def predict(data: PredictionRequest):
    X = np.array(data.features).reshape(1, -1)

    if model_type == "pkl":
        prediction = model.predict(X)
        result = prediction.tolist()
    else:
        prediction = model.predict(X)
        result = np.argmax(prediction, axis=1).tolist()

    return {
        "model_type": model_type,
        "prediction": result
    }

