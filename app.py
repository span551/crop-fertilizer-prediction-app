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
# WEATHER API (ONLY TEMP USED)
# -------------------------------
API_KEY = "8309fcb8b95494c5b9bc81eda528ae31"

def get_temperature(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            return data['main']['temp']
        else:
            return None
    except:
        return None

# -------------------------------
# 🌧 SEASONAL RAINFALL DATA
# -------------------------------
rainfall_data = {
    "Nagpur": {"Kharif": 900, "Rabi": 100, "Zaid": 50},
    "Delhi": {"Kharif": 600, "Rabi": 50, "Zaid": 20},
    "Pune": {"Kharif": 700, "Rabi": 80, "Zaid": 40},
    "Bangalore": {"Kharif": 700, "Rabi": 150, "Zaid": 80},
    "Mumbai": {"Kharif": 2000, "Rabi": 50, "Zaid": 20},
    "Chennai": {"Kharif": 800, "Rabi": 300, "Zaid": 100},
    "Kolkata": {"Kharif": 1200, "Rabi": 100, "Zaid": 60},
    "Bhopal": {"Kharif": 850, "Rabi": 70, "Zaid": 30},
    "Hyderabad": {"Kharif": 750, "Rabi": 60, "Zaid": 40},
    "Ahmedabad": {"Kharif": 500, "Rabi": 40, "Zaid": 20},
    "Jaipur": {"Kharif": 400, "Rabi": 30, "Zaid": 10},
    "Lucknow": {"Kharif": 900, "Rabi": 100, "Zaid": 50}
}

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="header">🌾 Smart Agriculture Assistant</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------
# CITY + SEASON
# -------------------------------
st.subheader("📍 Location & Season")

city = st.selectbox("Select City", list(rainfall_data.keys()))
season = st.selectbox("Select Season", ["Kharif", "Rabi", "Zaid"])

# Fetch temperature
if st.button("Fetch Temperature"):
    temp = get_temperature(city)
    if temp:
        st.session_state["temp"] = temp
        st.success(f"🌡 Temperature: {temp}°C")
    else:
        st.error("❌ Could not fetch temperature")

# -------------------------------
# INPUTS
# -------------------------------
st.markdown("---")
st.subheader("Enter Soil Details")

col1, col2, col3 = st.columns(3)

with col1:
    nitrogen = st.number_input("Nitrogen", 0, 400, 50)
with col2:
    phosphorus = st.number_input("Phosphorus", 0, 400, 40)
with col3:
    potassium = st.number_input("Potassium", 0, 400, 35)

col4, col5, col6 = st.columns(3)

with col4:
    temperature = st.number_input("Temperature", 0.0, 50.0, float(st.session_state.get("temp", 25.0)))

# 🌧 Smart Rainfall
seasonal_rainfall = rainfall_data.get(city, {}).get(season, 100)
rainfall = seasonal_rainfall

with col5:
    st.number_input("Rainfall (auto)", value=rainfall, disabled=True)

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

    st.warning("⚠️ These predictions are based on dataset patterns and may vary in real-world agricultural conditions.")

    features = np.array([[nitrogen, phosphorus, potassium, ph, rainfall, temperature]])

    probs = crop_model.predict_proba(features)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]

    top3_crops = le_crop.inverse_transform(top3_idx)
    top3_probs = probs[top3_idx]

    fert_pred = fert_model.predict(features)
    fert_name = le_fert.inverse_transform(fert_pred)[0]

    st.markdown("---")

    # 🌱 Crop
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

    # 🧠 Explainability
    st.subheader("🧠 Why this recommendation?")
    if rainfall > 800:
        st.write("✔ High rainfall supports crops like rice & sugarcane")
    if 20 < temperature < 35:
        st.write("✔ Optimal temperature detected")
    if 6 < ph < 7.5:
        st.write("✔ Soil pH is ideal")

    # 🧪 Suggestions
    st.subheader("🧪 Soil Suggestions")
    if nitrogen < 40:
        st.warning("Low Nitrogen → Add Urea")
    if phosphorus < 40:
        st.warning("Low Phosphorus → Add DAP")
    if potassium < 40:
        st.warning("Low Potassium → Add MOP")
    if ph < 5.5:
        st.warning("Acidic soil → Add Lime")
    if ph > 7.5:
        st.warning("Alkaline soil → Add Gypsum")
