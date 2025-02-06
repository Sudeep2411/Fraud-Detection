import streamlit as st
import pandas as pd
from data_preprocessing import load_data, preprocess_data
from model_training import train_model
from real_time_simulation import simulate_transactions

# Streamlit app
st.title("Real-Time Fraud Detection System")

# 1. Load Dataset
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 2. Preprocessing
@st.cache
def preprocess_data(df):
    X_resampled, y_resampled, scaler = preprocess_data(df)
    return X_resampled, y_resampled, scaler

# 3. Train Model
@st.cache
def train_model(X_resampled, y_resampled):
    model, accuracy, report, X_test = train_model(X_resampled, y_resampled)
    return model, accuracy, report, X_test

# Streamlit interface
st.header("Dataset Overview")
file_path = "creditcard.csv"
df = load_data(file_path)
st.write(df.head())

# Preprocessing
st.header("Data Preprocessing")
X_resampled, y_resampled, scaler = preprocess_data(df)

# Model Training
st.header("Model Training")
model, accuracy, report, X_test = train_model(X_resampled, y_resampled)
st.write(f"Model Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(report)

# Real-Time Fraud Detection Simulation
st.header("Real-Time Fraud Detection Simulation")
if st.button("Start Simulation"):
    with st.spinner("Simulating transactions..."):
        for index, is_fraud in simulate_transactions(model, scaler, pd.DataFrame(X_test[:10])):
            st.write(f"Transaction {index + 1}: {is_fraud}")
