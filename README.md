Real-Time Fraud Detection System

This project implements a real-time fraud detection system using machine learning techniques. It processes a credit card transactions dataset, scales the features, and addresses class imbalance to train a fraud detection model. The system includes a real-time transaction simulation and an interactive web-based dashboard.

🚀 Quick Start

1. Clone the Repository

git clone https://github.com/username/fraud_detection_project.git
cd fraud_detection_project

2. Install Dependencies

Ensure you have Python 3.8 or higher and install the required libraries:

pip install -r requirements.txt

3. Run the Streamlit App

streamlit run app.py

🔹 Features

Data Preprocessing: Scaling features and handling class imbalance using SMOTE.

Machine Learning Model: Building and training a fraud detection model.

Real-Time Simulation: Simulating real-time fraud predictions.

Interactive Dashboard: A web-based visualization using Streamlit.

📂 Directory Structure

fraud_detection_project/
├── creditcard.csv             # Dataset file
├── data_preprocessing.py      # Preprocessing script
├── model_training.py         # Model training script
├── real_time_simulation.py   # Real-time simulation script
├── app.py                    # Streamlit app
├── README.md                 # Project documentation
├── venv/                     # Virtual environment (optional)

📊 Dataset

The dataset (creditcard.csv) should be placed in the root directory. If unavailable, download it from Kaggle.

⚙️ Usage

1️⃣ Data Preprocessing

python data_preprocessing.py

2️⃣ Model Training

python model_training.py

3️⃣ Real-Time Fraud Simulation

python real_time_simulation.py

4️⃣ Web Application

streamlit run app.py

📌 Output

Preprocessed Dataset: Scaled features and balanced classes.

Trained Model: A saved fraud detection model.

Real-Time Predictions: Fraud prediction results for transactions.

Interactive Dashboard: A user-friendly interface for insights.

🔮 Future Enhancements

Improve model efficiency for large-scale datasets.

Implement real-time alerts for flagged transactions.

Enhance Streamlit dashboard with advanced visualizations.

Explore deep learning models for better fraud detection accuracy.

📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

🙌 Acknowledgments

This project uses the Kaggle Credit Card Fraud Dataset.


