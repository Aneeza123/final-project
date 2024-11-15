import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt

# Function to load the dataset
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"File {file_path} not found!")
        return None

# Load the dataset
df = load_data('exercise_angles.csv')  # Make sure this file is in the same directory as the app

if df is not None:
    # Preprocessing
    st.write("### Dataset Overview:")
    st.write(df.head())

    # Check the distribution of exercises
    st.write("### Exercise Type Distribution:")
    exercise_counts = df['Label'].value_counts()
    st.write(exercise_counts)

    # Plot the distribution
    fig, ax = plt.subplots()
    exercise_counts.plot(kind='bar', ax=ax)
    ax.set_title('Exercise Type Distribution')
    ax.set_xlabel('Exercise Type')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Handle missing values by dropping rows with NaN values
    df.dropna(inplace=True)

    # Encode the target variable 'Label' (Exercise type)
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])

    # Define features (X) and target (y)
    X = df[['Shoulder_Angle', 'Elbow_Angle', 'Hip_Angle', 'Knee_Angle', 'Ankle_Angle',
            'Shoulder_Ground_Angle', 'Elbow_Ground_Angle', 'Hip_Ground_Angle',
            'Knee_Ground_Angle', 'Ankle_Ground_Angle']]
    y = df['Label']

    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Normalize/Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = rf_model.predict(X_test_scaled)
    st.write("### Model Evaluation Report:")
    st.text(classification_report(y_test, y_pred))

    # Streamlit interface
    st.title('Exercise Type Prediction based on Angles')

    # Input fields for angles (use sliders)
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

    # Collect all input angles into a DataFrame
    input_data = np.array([[shoulder_angle, elbow_angle, hip_angle, knee_angle, ankle_angle,
                            shoulder_ground_angle, elbow_ground_angle, hip_ground_angle,
                            knee_ground_angle, ankle_ground_angle]])

    # Scale the input data using the same scaler
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = rf_model.predict(input_data_scaled)

    # Inverse transform the predicted label to get the exercise type name
    predicted_label = label_encoder.inverse_transform(prediction)

    # Display the predicted exercise type
    st.write(f'Predicted Exercise Type: {predicted_label[0]}')

else:
    st.error("Dataset not found or unable to load!")
