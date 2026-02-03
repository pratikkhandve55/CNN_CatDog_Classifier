import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Page config
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

st.title("🐱🐶 Cat vs Dog Image Classifier")
st.caption("Upload an image to get real-time CNN prediction")

# Load model
model = load_model('model/cnn_model.h5')

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(
            uploaded_file,
            width=220,              # ✅ Reduced image size
            caption="Input Image"
        )

    # Preprocess


    img = Image.open(uploaded_file).convert("RGB")  # FORCE 3 channels
    img = img.resize((64, 64))                    # SAME as training size

    img_array = np.array(img) / 255.0                # Normalize
    img_array = np.expand_dims(img_array, axis=0)    # Shape: (1, 128, 128, 3)

    prediction = model.predict(img_array)[0][0]


    with col2:
        st.subheader("🔍 Prediction")

        if prediction > 0.5:
            confidence = prediction * 100
            st.success("🐶 Dog Detected")
        else:
            confidence = (1 - prediction) * 100
            st.success("🐱 Cat Detected")

        # ✅ Confidence bar
        st.progress(int(confidence))

        # Confidence text
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")

        # Friendly feedback
        if confidence > 85:
            st.caption("🟢 Very confident prediction")
        elif confidence > 65:
            st.caption("🟡 Moderate confidence")
        else:
            st.caption("🔴 Low confidence")
