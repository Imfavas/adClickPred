import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load your pre-trained model, feature columns, and scaler
with open("ad_click.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("feature_columns.pkl", "rb") as columns_file:
    feature_columns = pickle.load(columns_file)  # List of feature names used during training

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)  # StandardScaler instance used during training

# Title
st.title("Ad Click Prediction")

# User Inputs
st.header("Provide User Information")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, step=1, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary"])
device_type = st.selectbox("Device Type", ["Desktop", "Mobile", "Tablet"])
ad_position = st.selectbox("Ad Position", ["Top", "Side", "Bottom"])
browsing_history = st.selectbox(
    "Browsing History",["Shopping", "Education", "Entertainment", "Social Media", "News"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

# Prediction Logic
if st.button("Predict"):
    # Create a dictionary to match the user input to feature columns
    input_data = {
        "age": age,
        "gender_Male": 1 if gender == "Male" else 0,
        "gender_Non-Binary": 1 if gender == "Non-Binary" else 0,
        "device_type_Mobile": 1 if device_type == "Mobile" else 0,
        "device_type_Tablet": 1 if device_type == "Tablet" else 0,
        "ad_position_Side": 1 if ad_position == "Side" else 0,
        "ad_position_Bottom": 1 if ad_position == "Bottom" else 0,
        "time_of_day_Afternoon": 1 if time_of_day == "Afternoon" else 0,
        "time_of_day_Evening": 1 if time_of_day == "Evening" else 0,
        "time_of_day_Night": 1 if time_of_day == "Night" else 0,
    }

    # Handle browsing history
    for category in ["Shopping", "Education", "Entertainment", "Social Media", "News"]:
        input_data[f"browsing_history_{category}"] = 1 if category in browsing_history else 0

    # Ensure all columns are present in the same order as feature_columns
    input_df = pd.DataFrame([input_data], columns=feature_columns).fillna(0)

    # Scale the input data using the pre-loaded scaler
    scaled_input = scaler.transform(input_df)

    # Predict using the model
    prediction = model.predict(scaled_input)

    # Interpret the result
    result = "Click" if prediction[0] == 1 else "No Click"
    st.write(f"Prediction: {result}")

# Footer
st.write("---")
