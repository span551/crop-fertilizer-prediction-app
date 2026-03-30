import streamlit as st  
import numpy as np
import pickle
import requests
import plotly.graph_objects as go

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="MahaKrishi AI", layout="centered")

# -------------------------------
# CSS
# -------------------------------
st.markdown("""
<style>

body {
    background-color: #f5f7fa;
}

.header {
    background: linear-gradient(90deg, #1b5e20, #43a047);
    padding: 18px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

.card {
    background: white;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.top-card {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    padding: 18px;
    border-radius: 16px;
    border-left: 6px solid #2e7d32;
}

.chat-bubble {
    background: #e8f5e9;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 8px;
}

div.stButton > button {
    background: linear-gradient(90deg, #2e7d32, #66bb6a);
    color: white;
    border-radius: 25px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
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
# 🌾 YIELD ESTIMATION FUNCTION
# -------------------------------
def estimate_yield(crop, n, p, k, ph, rainfall, temp):

    base_yield = {
        "rice": 20,
        "wheat": 18,
        "maize": 22,
        "sugarcane": 350,
        "cotton": 10,
        "millet": 12,
        "pulses": 8
    }

    crop = crop.lower()
    base = base_yield.get(crop, 15)

    nutrient_score = (n + p + k) / 300

    if 6 <= ph <= 7.5:
        ph_factor = 1
    else:
        ph_factor = 0.8

    if rainfall > 800:
        rain_factor = 1
    elif rainfall > 400:
        rain_factor = 0.8
    else:
        rain_factor = 0.6

    if 20 <= temp <= 35:
        temp_factor = 1
    else:
        temp_factor = 0.85

    yield_estimate = base * nutrient_score * ph_factor * rain_factor * temp_factor

    return round(yield_estimate, 2)

# -------------------------------
# WEATHER API
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
# RAINFALL DATA
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

# -------------------------------
# LOCATION
# -------------------------------
st.subheader("📍 Location & Season")

city = st.selectbox("Select City", list(rainfall_data.keys()))
season = st.selectbox("Select Season", ["Kharif", "Rabi", "Zaid"])

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
st.markdown("### 🌱 Soil Details")

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

rainfall = rainfall_data.get(city, {}).get(season, 100)

with col5:
    st.number_input("Rainfall", value=rainfall, disabled=True)

with col6:
    ph = st.number_input("pH", 0.0, 14.0, 6.5)

# -------------------------------
# BUTTON
# -------------------------------
predict_btn = st.button("🚀 Get Recommendation")

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

    # 🌾 BEST CROP
    st.markdown("### 🌾 Best Crop")
    st.markdown(f"""
    <div class="top-card">
        <h2>🥇 {top3_crops[0]}</h2>
        <p>Confidence: {top3_probs[0]*100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # 🎯 CONFIDENCE GAUGE
    confidence = top3_probs[0] * 100
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "AI Confidence"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 📊 BAR CHART
    fig = go.Figure(go.Bar(
        x=top3_probs,
        y=top3_crops,
        orientation='h',
        text=[f"{p*100:.1f}%" for p in top3_probs],
        textposition='auto'
    ))
    st.plotly_chart(fig, use_container_width=True)

    # 🧪 FERTILIZER
    st.markdown(f"""
    <div class="card">
        <h3>🧪 Fertilizer Recommendation</h3>
        <h2 style="color:#2e7d32;">{fert_name}</h2>
    </div>
    """, unsafe_allow_html=True)

    # 💧 IRRIGATION
    st.markdown("### 💧 Irrigation Advice")
    if rainfall < 100:
        st.error("Low rainfall → Use irrigation")
    elif rainfall < 300:
        st.warning("Moderate rainfall → Consider irrigation")
    else:
        st.success("Sufficient rainfall")

    # 🤖 AI CHAT
    st.markdown("### 🤖 AI Assistant")

    if nitrogen < 40:
        st.markdown('<div class="chat-bubble">🌿 Nitrogen is low → Add Urea</div>', unsafe_allow_html=True)

    if rainfall < 100:
        st.markdown('<div class="chat-bubble">💧 Use drip irrigation</div>', unsafe_allow_html=True)

    if 6 < ph < 7.5:
        st.markdown('<div class="chat-bubble">🧪 Soil pH is optimal</div>', unsafe_allow_html=True)

    # 🚜 YIELD
    st.markdown("### 🚜 Expected Yield (Per Acre)")

    predicted_yield = estimate_yield(
        top3_crops[0],
        nitrogen,
        phosphorus,
        potassium,
        ph,
        rainfall,
        temperature
    )

    st.markdown(f"""
    <div class="card">
        <h2>🌾 {predicted_yield} Quintals / Acre</h2>
        <p>Estimated yield for <b>{top3_crops[0]}</b></p>
    </div>
    """, unsafe_allow_html=True)

    if predicted_yield > 20:
        st.success("🔥 High yield expected!")
    elif predicted_yield > 10:
        st.warning("⚖️ Moderate yield expected")
    else:
        st.error("⚠️ Low yield — improve soil or irrigation")

    # 🔍 WHY NOT
    st.markdown("### 🔍 Why not other crops?")
    for i in range(1, 3):
        st.markdown(f"""
        <div class="card">
        ❌ {top3_crops[i]} has lower suitability ({top3_probs[i]*100:.2f}%)
        </div>
        """, unsafe_allow_html=True)
