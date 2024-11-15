import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the pre-trained model
model = joblib.load('exercise_model.pkl')

# Streamlit UI for input
st.title('Exercise Prediction App')

st.write("Enter the angles of the body parts:")

# Input fields for angles
shoulder_angle = st.number_input('Shoulder Angle', min_value=0, max_value=180)
elbow_angle = st.number_input('Elbow Angle', min_value=0, max_value=180)
hip_angle = st.number_input('Hip Angle', min_value=0, max_value=180)
knee_angle = st.number_input('Knee Angle', min_value=0, max_value=180)
ankle_angle = st.number_input('Ankle Angle', min_value=0, max_value=180)

# Prediction button
if st.button('Predict'):
    input_data = np.array([[shoulder_angle, elbow_angle, hip_angle, knee_angle, ankle_angle]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the result
    st.write(f"Predicted Exercise: {prediction[0]}")
