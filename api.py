from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import io
import os

app = FastAPI(title="Crop Disease Prediction API")

# ------------------ CORS CONFIGURATION ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

# ------------------ DATASTORE (Placeholder) ------------------
# Replace with actual disease data mapping
DISEASE_INFO = {
    1: {
        "disease": "Healthy",
        "remedies": ["Keep maintaining good care."],
        "pesticides": ["None needed."],
        "prevention": ["Regular monitoring."]
    },
    0: {
        "disease": "Rust",
        "remedies": ["Remove infected leaves.", "Apply fungicides."],
        "pesticides": ["Sulfur dust", "Mancozeb"],
        "prevention": ["Crop rotation", "Resistant varieties."]
    },
    2: {
        "disease": "Powdery Mildew",
        "remedies": ["Neem oil spray", "Milk spray mixture"],
        "pesticides": ["Potassium bicarbonate", "Sulfur"],
        "prevention": ["Proper spacing", "Sunlight exposure"]
    },
    # Add more classes as per your model's output classes
}

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

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Read and process image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224))
    
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)
    
    if prediction.ndim > 1 and prediction.shape[1] > 1:
        class_index = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
    else:
        # Fallback for binary or single output
        class_index = int(np.round(prediction)[0]) if prediction.shape[0] > 0 else 0
        confidence = float(prediction[0]) if prediction.shape[0] > 0 else 0.0

    # Retrieve disease info
    info = DISEASE_INFO.get(class_index, {
        "disease": f"Unknown Class {class_index}",
        "remedies": [],
        "pesticides": [],
        "prevention": []
    })

    return {
        "class_index": class_index,
        "confidence": confidence,
        "disease": info["disease"],
        "remedies": info["remedies"],
        "pesticides": info["pesticides"],
        "prevention": info["prevention"]
    }

