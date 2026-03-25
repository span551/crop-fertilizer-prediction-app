import streamlit as st
import numpy as np
import pickle
import requests

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Agri AI", layout="centered")

# -------------------------------
# CUSTOM CSS (UI DESIGN)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7f6;
}
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
st.markdown('<div class="header">🌾 Crop and Fertilizer Recommendation System</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------
# WEATHER INPUT
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
        st.error("❌ Could not fetch weather data")

st.markdown("---")

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("Enter Details")

col1, col2, col3 = st.columns(3)

with col1:
    nitrogen = st.number_input("Nitrogen (N)", 0, 140, 50)

with col2:
    phosphorus = st.number_input("Phosphorus (P)", 0, 140, 40)

with col3:
    potassium = st.number_input("Potassium (K)", 0, 140, 35)

col4, col5, col6 = st.columns(3)

with col4:
    temperature = st.number_input(
        "Temperature (°C)", 
        0.0, 50.0, 
        float(st.session_state.get("temp", 25.0))
    )

with col5:
    rainfall = st.number_input(
        "Rainfall (mm)", 
        0.0, 300.0, 
        float(st.session_state.get("rain", 100.0))
    )

with col6:
    ph = st.number_input("pH Level", 0.0, 14.0, 6.5)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)

center_col = st.columns([1,20,1])
with center_col[1]:
    predict_btn = st.button("Predict")

# -------------------------------
# PREDICTION OUTPUT
# -------------------------------
if predict_btn:
    try:
        features = np.array([[nitrogen, phosphorus, potassium, ph, rainfall, temperature]])

        crop_pred = crop_model.predict(features)
        fert_pred = fert_model.predict(features)

        crop_name = le_crop.inverse_transform(crop_pred)[0]
        fert_name = le_fert.inverse_transform(fert_pred)[0]

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # Crop Card
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🌱 Recommended Crop")
            
            st.markdown(f"### {crop_name}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Fertilizer Card
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🧪 Fertilizer Recommendation")
            st.write(f"✔ {fert_name}")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
