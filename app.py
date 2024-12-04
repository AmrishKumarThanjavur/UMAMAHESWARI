import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from PIL import Image
import pickle

# Load dataset and handle errors
try:
    dataset = pd.read_csv("datasets/general.csv")
    X = dataset.drop('Disease', axis=1)
    y = dataset['Disease']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
except FileNotFoundError as e:
    st.error("Dataset file not found. Please check the path.")
    X, y = None, None

# Train model (for local testing; this should be saved/reused)
model = RandomForestClassifier(random_state=42)
if X is not None and y is not None:
    model.fit(X, y)

# Load pre-trained models
try:
    diabetes_model = pickle.load(open('Models/Diabetes.sav', 'rb'))
    heart_disease_model = pickle.load(open('Models/General.sav', 'rb'))
    parkinsons_model = pickle.load(open('Models/Parkinsons.sav', 'rb'))
except FileNotFoundError:
    st.error("One or more model files are missing. Ensure all files are in the correct directory.")

# Utility functions
def predict_disease(temp_f, pulse_rate_bpm, vomiting, yellowish_urine, indigestion):
    try:
        user_input = pd.DataFrame({
            'Temp': [float(temp_f)],
            'Pulserate': [float(pulse_rate_bpm)],
            'Vomiting': [int(vomiting)],
            'YellowishUrine': [int(yellowish_urine)],
            'Indigestion': [int(indigestion)]
        })
        user_input = scaler.transform(user_input)
        predicted_disease = model.predict(user_input)[0]
        disease_names = {0: 'Heart Disease', 1: 'Viral Fever/Cold', 2: 'Jaundice', 3: 'Food Poisoning', 4: 'Normal'}
        return disease_names.get(predicted_disease, "Unknown Disease")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return "Error"

def calculate_bmi(weight, height):
    try:
        bmi = float(weight) / (float(height) / 100) ** 2
        return bmi
    except ValueError:
        st.error("Invalid input. Ensure weight and height are numbers.")
        return None

def interpret_bmi(bmi):
    if bmi is None:
        return "Invalid BMI"
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal Weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

# Main application
def main():
    st.sidebar.image('images/navbar.png', width=200)
    selected = option_menu(
        'Disease Diagnosis and Recommendation System',
        ['GENERAL', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'BMI CALCULATOR'],
        icons=['dashboard', 'activity', 'heart', 'person', 'line-chart'],
        default_index=0
    )

    if selected == 'GENERAL':
        st.title("General Diagnosis")
        temp_f = st.text_input("Temperature (F):")
        pulse_rate_bpm = st.text_input("Pulse Rate (bpm):")
        vomiting = st.checkbox("Vomiting")
        yellowish_urine = st.checkbox("Yellowish Urine")
        indigestion = st.checkbox("Indigestion")

        if st.button("Test Result"):
            prediction = predict_disease(temp_f, pulse_rate_bpm, vomiting, yellowish_urine, indigestion)
            st.info(f"Predicted Disease: {prediction}")

    # Other sections (Diabetes, Heart Disease, Parkinson's, BMI) can be implemented similarly

if __name__ == "__main__":
    main()
