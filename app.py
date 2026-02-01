import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import os

st.set_page_config(
    page_title="Crop Disease Prediction",
    page_icon="🌱",
    layout="centered"
)

st.title("🌾 Crop Disease Prediction")
st.write("Predict crop disease using a trained ML/DL model")

# ------------------ MODEL LOADING ------------------

@st.cache_resource
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

if model is None:
    st.error("❌ No model file found (.pkl / .h5 / .keras)")
    st.stop()

st.success(f"✅ Model loaded successfully ({model_type})")

# ------------------ INPUT SECTION ------------------

st.subheader("🔢 Enter Input Features")

num_features = st.number_input(
    "Number of input features",
    min_value=1,
    max_value=50,
    value=4
)

inputs = []
for i in range(num_features):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

# ------------------ PREDICTION ------------------

if st.button("🚀 Predict"):
    X = np.array(inputs).reshape(1, -1)

    if model_type == "pkl":
        prediction = model.predict(X)
    else:
        prediction = model.predict(X)
        prediction = np.argmax(prediction, axis=1)

    st.subheader("🧪 Prediction Result")
    st.write(prediction)
