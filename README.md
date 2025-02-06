Real-Time Fraud Detection System

This project implements a real-time fraud detection system using machine learning techniques. It processes a credit card transactions dataset, scales the features, and handles class imbalance to prepare the data for training a fraud detection model.

Quick Start

Clone the repository: git clone https://github.com/username/fraud_detection_project.git

Navigate to the directory: cd fraud_detection_project

Install dependencies: pip install -r requirements.txt

Run the Streamlit app: streamlit run app.py

Features

Data Preprocessing: Scaling features and handling class imbalance using SMOTE.

Machine Learning: Building and training a fraud detection model.

Real-Time Simulation: Simulating real-time predictions of fraudulent transactions.

Web Application: An interactive web-based dashboard using Streamlit.

Dataset

Place the dataset creditcard.csv in the root directory of the project.If not available locally, download it here.

Requirements

Python 3.8 or higher

Pip (latest version)

Install required libraries using pip:

pip install pandas scikit-learn imbalanced-learn streamlit

Directory Structure

fraud_detection_project/
├── creditcard.csv             # Dataset file
├── data_preprocessing.py      # Preprocessing script
├── model_training.py          # Training script
├── real_time_simulation.py    # Real-time simulation script
├── app.py                     # Streamlit app
├── README.md                  # Project documentation
├── venv/                      # Virtual environment

Usage

Data Preprocessing:Run the preprocessing script to load and preprocess the dataset:

python data_preprocessing.py

Model Training:Train the machine learning model:

python model_training.py

Real-Time Simulation:Simulate real-time predictions:

python real_time_simulation.py

Web Application:Run the Streamlit app to interact with the fraud detection system:

streamlit run app.py

Output

Preprocessed Dataset: Scaled features and balanced classes.

Trained Model: A saved fraud detection model.

Real-Time Predictions: Fraud prediction results for transactions.

Interactive Dashboard: A user-friendly dashboard for fraud detection insights.

Future Enhancements

Optimize the model for better performance on large-scale datasets.

Add real-time alerts for flagged transactions.

Implement advanced visualizations in the Streamlit app.

Explore deep learning models for better detection accuracy.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Kaggle Credit Card Fraud Dataset

