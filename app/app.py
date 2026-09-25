import sys
import os
from datetime import datetime, date, time

import streamlit as st
import pandas as pd

# Allow imports from src/ when the app is launched as `streamlit run app/app.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.save_model import load_model, DEFAULT_MODEL_PATH
from src.predict import build_input_row, predict_rentals

st.set_page_config(page_title="Bike Rental Demand Predictor", page_icon="🚲", layout="centered")

st.title("🚲 Bike Rental Demand Predictor")
st.write(
    "Estimate hourly Capital Bikeshare rental demand from weather and calendar conditions."
)


@st.cache_resource
def get_model():
    if not os.path.exists(DEFAULT_MODEL_PATH):
        return None
    return load_model(DEFAULT_MODEL_PATH)


model = get_model()

if model is None:
    st.error(
        f"No trained model found at `{DEFAULT_MODEL_PATH}`. "
        "Run `python -m src.model_comparison` first to train and save a model."
    )
    st.stop()

st.sidebar.header("Date & Time")
input_date = st.sidebar.date_input("Date", value=date(2012, 6, 15))
input_time = st.sidebar.time_input("Time", value=time(8, 0))

st.sidebar.header("Calendar")
season = st.sidebar.selectbox(
    "Season", options=[1, 2, 3, 4], format_func=lambda s: {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}[s]
)
holiday = st.sidebar.radio("Holiday?", options=[0, 1], format_func=lambda v: "Yes" if v else "No", horizontal=True)
workingday = st.sidebar.radio(
    "Working day?", options=[0, 1], format_func=lambda v: "Yes" if v else "No", horizontal=True
)

st.sidebar.header("Weather")
weather = st.sidebar.selectbox(
    "Condition",
    options=[1, 2, 3, 4],
    format_func=lambda w: {
        1: "Clear / Few clouds",
        2: "Mist / Cloudy",
        3: "Light Snow / Light Rain",
        4: "Heavy Rain / Ice / Thunderstorm",
    }[w],
)
temp = st.sidebar.slider("Temperature (°C)", min_value=0.0, max_value=45.0, value=22.0, step=0.5)
atemp = st.sidebar.slider("Feels-like temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=55)
windspeed = st.sidebar.slider("Windspeed", min_value=0.0, max_value=60.0, value=10.0, step=0.5)

if st.sidebar.button("Predict demand", type="primary"):
    dt = pd.Timestamp(datetime.combine(input_date, input_time))
    input_row = build_input_row(
        dt=dt,
        season=season,
        holiday=holiday,
        workingday=workingday,
        weather=weather,
        temp=temp,
        atemp=atemp,
        humidity=humidity,
        windspeed=windspeed,
    )

    prediction = predict_rentals(model, input_row)

    st.metric("Predicted rentals this hour", f"{prediction:.0f}")
    with st.expander("Model input"):
        st.dataframe(input_row)
else:
    st.info("Set the conditions in the sidebar, then click **Predict demand**.")
