import streamlit as st
import pickle
import numpy as np

with open("trained_model.sav", "rb") as f:
    model, scaler = pickle.load(f)

st.title("Car MPG Predictor")

st.write("Enter car details below to predict the **Miles per Gallon (MPG)**.")

cylinders = st.number_input("Cylinders", min_value=3, max_value=12, value=4)
displacement = st.number_input("Displacement (cc)", min_value=50.0, max_value=800.0, value=150.0)
horsepower = st.number_input("Horsepower", min_value=40.0, max_value=300.0, value=100.0)
weight = st.number_input("Weight (lbs)", min_value=1000.0, max_value=6000.0, value=2500.0)
acceleration = st.number_input("Acceleration (0-60 mph in seconds)", min_value=5.0, max_value=30.0, value=15.0)
origin = st.selectbox("Origin", options=[1, 2, 3], format_func=lambda x: {1:"USA", 2:"Europe", 3:"Japan"}[x])

features = np.array([[cylinders, displacement, horsepower, weight,acceleration,origin]])

features_scaled = scaler.transform(features)

if st.button("Predict MPG"):
    prediction = model.predict(features_scaled)
    st.success(f"Estimated MPG: {prediction[0]:.2f}")

st.set_page_config(page_title="Car MPG Predictor", layout="centered")

try:
    with open("trained_model.sav", "rb") as f:
        model, scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
