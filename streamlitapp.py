import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import io

# Function to load the dataset from file upload
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        st.error("Please upload the dataset!")
        return None

# Streamlit interface
st.title('Exercise Type Prediction based on Angles')

# Upload file for prediction
uploaded_file = st.file_uploader("Upload Exercise Data", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None:
        # Preprocessing
        df.dropna(inplace=True)  # Handle missing values

        # Encode the target variable 'Label' (Exercise type)
        label_encoder = LabelEncoder()
        df['Label'] = label_encoder.fit_transform(df['Label'])

        # Define features (X) and target (y)
        X = df[['Shoulder_Angle', 'Elbow_Angle', 'Hip_Angle', 'Knee_Angle', 'Ankle_Angle',
                'Shoulder_Ground_Angle', 'Elbow_Ground_Angle', 'Hip_Ground_Angle',
                'Knee_Ground_Angle', 'Ankle_Ground_Angle']]
        y = df['Label']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize/Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize the RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the model
        rf_model.fit(X_train_scaled, y_train)

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

        # Scale the input data using the same scaler
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = rf_model.predict(input_data_scaled)

        # Inverse transform the predicted label to get the exercise type name
        predicted_label = label_encoder.inverse_transform(prediction)

        # Display the predicted label
        st.write(f'Predicted Exercise Type: {predicted_label[0]}')

else:
    st.warning("Please upload a CSV file to make a prediction!")
