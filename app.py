import streamlit as st
import numpy as np
import pickle

st.write("🚀 App started")

# Try loading files
try:
    crop_model = pickle.load(open("crop_model.pkl", "rb"))
    st.write("✅ crop_model loaded")

    fert_model = pickle.load(open("fert_model.pkl", "rb"))
    st.write("✅ fert_model loaded")

    le_crop = pickle.load(open("crop_encoder.pkl", "rb"))
    st.write("✅ crop_encoder loaded")

    le_fert = pickle.load(open("fert_encoder.pkl", "rb"))
    st.write("✅ fert_encoder loaded")

except Exception as e:
    st.error(f"❌ Error loading files: {e}")

# UI
st.title("🌾 Smart Agriculture Assistant")

nitrogen = st.slider("Nitrogen", 0, 140, 50)
phosphorus = st.slider("Phosphorus", 0, 140, 50)
potassium = st.slider("Potassium", 0, 140, 50)
ph = st.slider("pH", 0.0, 14.0, 6.5)
rainfall = st.slider("Rainfall", 0.0, 300.0, 100.0)
temperature = st.slider("Temperature", 0.0, 50.0, 25.0)

if st.button("Predict"):
    try:
        features = np.array([[nitrogen, phosphorus, potassium, ph, rainfall, temperature]])

        crop_pred = crop_model.predict(features)
        fert_pred = fert_model.predict(features)

        crop_name = le_crop.inverse_transform(crop_pred)
        fert_name = le_fert.inverse_transform(fert_pred)

        st.success(f"🌱 Crop: {crop_name[0]}")
        st.success(f"🧪 Fertilizer: {fert_name[0]}")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")