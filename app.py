import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
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
    """Load the first available model.

    Prefer image-based Keras models (.keras / .h5) so that the UI
    defaults to image upload when an image model is present.
    """
    if os.path.exists("crop_disease_model.keras"):
        return tf.keras.models.load_model("crop_disease_model.keras"), "keras"
    elif os.path.exists("crop_disease_model.h5"):
        return tf.keras.models.load_model("crop_disease_model.h5"), "h5"
    elif os.path.exists("crop_disease_model.pkl"):
        return joblib.load("crop_disease_model.pkl"), "pkl"
    else:
        return None, None

model, model_type = load_model()

if model is None:
    st.error("❌ No model file found (.pkl / .h5 / .keras)")
    st.stop()

st.success(f"✅ Model loaded successfully ({model_type})")

# ------------------ INPUT & PREDICTION SECTION ------------------

# For scikit-learn `.pkl` models we keep the numeric-feature UI.
# For Keras/TF `.keras` / `.h5` models we switch to an image-upload UI.

if model_type == "pkl":
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

    if st.button("🚀 Predict"):
        X = np.array(inputs).reshape(1, -1)
        prediction = model.predict(X)

        st.subheader("🧪 Prediction Result")
        st.write(prediction)
else:
    st.subheader("📷 Upload Crop Image")
    uploaded_file = st.file_uploader(
        "Upload a leaf/crop image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        # NOTE: Adjust the target size to match how the model was trained.
        target_size = (224, 224)
        image_resized = image.resize(target_size)

        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, 3)

        if st.button("🚀 Predict from Image"):
            prediction = model.predict(img_array)

            # For classification models, take argmax as the predicted class index.
            if prediction.ndim > 1 and prediction.shape[1] > 1:
                prediction_class = np.argmax(prediction, axis=1)
            else:
                prediction_class = prediction

            st.subheader("🧪 Prediction Result")
            st.write(prediction_class)
