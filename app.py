import streamlit as st 
import numpy as np
import pickle
import requests
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Agri AI", layout="centered")

# -------------------------------
# CSS
# -------------------------------
st.markdown("""
<style>
.header {
    background-color: #2e7d32;
    padding: 15px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODELS
# -------------------------------
try:
    crop_model = pickle.load(open("crop_model.pkl", "rb"))
    fert_model = pickle.load(open("fert_model.pkl", "rb"))
    le_crop = pickle.load(open("crop_encoder.pkl", "rb"))
    le_fert = pickle.load(open("fert_encoder.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Error loading models: {e}")

# -------------------------------
# WEATHER API
# -------------------------------
API_KEY = "8309fcb8b95494c5b9bc81eda528ae31"

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            temp = data['main']['temp']
            rainfall = data.get('rain', {}).get('1h', 0)
            return temp, rainfall
        else:
            return None, None
    except:
        return None, None

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="header">🌾 Smart Agriculture Assistant</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------
# WEATHER
# -------------------------------
st.subheader("📍 Auto Weather Detection")

city = st.text_input("Enter your city")

if st.button("Fetch Weather"):
    temp, rain = get_weather(city)

    if temp is not None:
        st.session_state["temp"] = temp
        st.session_state["rain"] = rain
        st.success(f"🌡 Temperature: {temp}°C")
        st.success(f"🌧 Rainfall: {rain} mm")
    else:
        st.error("❌ Could not fetch weather")

st.markdown("---")

# -------------------------------
# INPUTS
# -------------------------------
st.subheader("Enter Soil & Weather Details")

col1, col2, col3 = st.columns(3)

with col1:
    nitrogen = st.number_input("Nitrogen", 0, 200, 50)
with col2:
    phosphorus = st.number_input("Phosphorus", 0, 200, 40)
with col3:
    potassium = st.number_input("Potassium", 0, 200, 35)

col4, col5, col6 = st.columns(3)

with col4:
    temperature = st.number_input("Temperature", 0.0, 50.0, float(st.session_state.get("temp", 25.0)))
with col5:
    rainfall = st.number_input("Rainfall", 0.0, 1000.0, float(st.session_state.get("rain", 100.0)))
with col6:
    ph = st.number_input("pH", 0.0, 14.0, 6.5)

# -------------------------------
# BUTTON
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)
center_col = st.columns([1,2,1])
with center_col[1]:
    predict_btn = st.button("Predict")

# -------------------------------
# PREDICTION
# -------------------------------
if predict_btn:
    features = np.array([[nitrogen, phosphorus, potassium, ph, rainfall, temperature]])

    probs = crop_model.predict_proba(features)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]

    top3_crops = le_crop.inverse_transform(top3_idx)
    top3_probs = probs[top3_idx]

    fert_pred = fert_model.predict(features)
    fert_name = le_fert.inverse_transform(fert_pred)[0]

    st.markdown("---")

    # 🌱 Crop Section
    st.subheader("🌱 Crop Recommendations")

    for i in range(3):
        st.write(f"{i+1}. {top3_crops[i]} — {top3_probs[i]*100:.2f}%")

    # 📊 Chart
    fig, ax = plt.subplots()
    ax.bar(top3_crops, top3_probs)
    ax.set_title("Confidence Scores")
    st.pyplot(fig)

    # 🧪 Fertilizer
    st.subheader("🧪 Fertilizer Recommendation")
    st.success(f"{fert_name}")

    # -------------------------------
    # 🧠 EXPLAINABILITY
    # -------------------------------
    st.subheader("🧠 Why this recommendation?")

    if rainfall > 150:
        st.write("✔ High rainfall supports water-intensive crops")
    if 20 < temperature < 35:
        st.write("✔ Temperature is optimal for crop growth")
    if 6 < ph < 7.5:
        st.write("✔ Soil pH is ideal")
    if nitrogen > 80:
        st.write("✔ High nitrogen supports leafy growth")

    # -------------------------------
    # 🧪 SOIL SUGGESTIONS
    # -------------------------------
    st.subheader("🧪 Soil Improvement Suggestions")

    if nitrogen < 40:
        st.warning("Low Nitrogen → Add Urea")
    if phosphorus < 40:
        st.warning("Low Phosphorus → Add DAP")
    if potassium < 40:
        st.warning("Low Potassium → Add MOP")
    if ph < 5.5:
        st.warning("Soil is acidic → Add Lime")
    if ph > 7.5:
        st.warning("Soil is alkaline → Add Gypsum")
