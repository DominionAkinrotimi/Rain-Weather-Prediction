import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("rain_prediction_model.pkl")

# Streamlit UI
st.title("🌧️ Rain Prediction App ☀️")
st.write("Enter basic weather conditions to predict if it will rain.")

# User-friendly inputs
temperature = st.number_input("Temperature (°C)", value=25.0, step=0.1)
humidity = st.number_input("Humidity (%)", value=50.0, step=1.0)
wind_speed = st.number_input("Wind Speed (km/h)", value=10.0, step=0.5)
precipitation = st.number_input("Precipitation (mm)", value=0.0, step=0.1)
weather_condition = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy", "Stormy"])

# Feature Engineering - Convert user inputs to match model inputs
weather_code_map = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Stormy": 3}
weather_code = weather_code_map[weather_condition]

# Estimate missing features using assumptions
daylight_duration = 43200 if weather_condition in ["Sunny", "Cloudy"] else 36000  # 12 hrs or 10 hrs
sunshine_duration = 35000 if weather_condition == "Sunny" else 15000  # More sun for Sunny
uv_index_max = 7 if weather_condition == "Sunny" else 3  # UV index is higher when sunny
precipitation_hours = 0 if precipitation == 0 else 3  # Assume rain lasts 3 hours if precipitation > 0

# Construct feature array for prediction
features = np.array([[weather_code, temperature, temperature-3, temperature+2, temperature-2,
                      daylight_duration, sunshine_duration, uv_index_max, uv_index_max-2, precipitation, 
                      precipitation, 0, precipitation_hours, wind_speed, wind_speed*1.5, 
                      180, 5.0, 2.0]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(features)[0]  # Predict (0 = No Rain, 1 = Rain)
    probability = model.predict_proba(features)[0][1] * 100  # Probability of rain

    if prediction == 1:
        st.success(f"🌧️ Yes, it might rain! ({probability:.2f}% confidence)")
    else:
        st.info(f"☀️ No, it won't rain. ({100 - probability:.2f}% confidence)")

st.write("⚡️")
