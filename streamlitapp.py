import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the trained model and scaler
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

scaler = StandardScaler()
scaler.fit(X[['Shoulder_Angle', 'Elbow_Angle', 'Hip_Angle', 'Knee_Angle', 'Ankle_Angle',
              'Shoulder_Ground_Angle', 'Elbow_Ground_Angle', 'Hip_Ground_Angle',
              'Knee_Ground_Angle', 'Ankle_Ground_Angle']])

label_encoder = LabelEncoder()
label_encoder.fit(df['Label'])

# Streamlit interface
st.title('Exercise Type Prediction based on Angles')

# Input fields for angles
shoulder_angle = st.slider('Shoulder Angle', min_value=0, max_value=180, value=90)
elbow_angle = st.slider('Elbow Angle', min_value=0, max_value=180, value=90)
hip_angle = st.slider('Hip Angle', min_value=0, max_value=180, value=90)
knee_angle = st.slider('Knee Angle', min_value=0, max_value=180, value=90)
ankle_angle = st.slider('Ankle Angle', min_value=-90, max_value=90, value=0)

shoulder_ground_angle = st.slider('Shoulder Ground Angle', min_value=-90, max_value=90, value=0)
elbow_ground_angle = st.slider('Elbow Ground Angle', min_value=-90, max_value=90, value=0)
hip_ground_angle = st.slider('Hip Ground Angle', min_value=-90, max_value=90, value=0)
knee_ground_angle = st.slider('Knee Ground Angle', min_value=-90, max_value=90, value=0)
ankle_ground_angle = st.slider('Ankle Ground Angle', min_value=-90, max_value=90, value=0)

# Collect all inputs into a DataFrame
input_data = np.array([[shoulder_angle, elbow_angle, hip_angle, knee_angle, ankle_angle,
                        shoulder_ground_angle, elbow_ground_angle, hip_ground_angle,
                        knee_ground_angle, ankle_ground_angle]])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = rf_model.predict(input_data_scaled)

# Display the predicted label
predicted_label = label_encoder.inverse_transform(prediction)
st.write(f'Predicted Exercise Type: {predicted_label[0]}')
