import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('exercise_model.pkl')

# Streamlit App Interface
st.title("Exercise Classification")
st.write("Enter the angles to classify the exercise.")

# Input fields for user input
shoulder_angle = st.number_input("Shoulder Angle", min_value=0, max_value=180, value=90)
elbow_angle = st.number_input("Elbow Angle", min_value=0, max_value=180, value=90)
hip_angle = st.number_input("Hip Angle", min_value=0, max_value=180, value=90)
knee_angle = st.number_input("Knee Angle", min_value=0, max_value=180, value=90)
ankle_angle = st.number_input("Ankle Angle", min_value=0, max_value=180, value=90)

# Predict button
if st.button('Predict'):
    # Prepare input for prediction
    input_features = np.array([[shoulder_angle, elbow_angle, hip_angle, knee_angle, ankle_angle]])
    prediction = model.predict(input_features)

    # Display prediction result
    st.write(f'Predicted Exercise: {prediction[0]}')
