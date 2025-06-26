
# 🛡️ Real-Time Fraud Detection System

A real-time fraud detection system that processes credit card transactions, balances imbalanced data using SMOTE, trains a machine learning model, and provides live predictions via a **Streamlit dashboard**.

---

## 📌 Project Highlights

| Feature                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| 🧹 **Data Preprocessing** | Handles missing values, feature scaling, and class balancing with SMOTE.     |
| 🤖 **Model Training**     | Uses Random Forest for classification and evaluates using accuracy/metrics. |
| 🕒 **Real-Time Simulation** | Simulates transactions with delay and displays fraud predictions.             |
| 📊 **Streamlit UI**       | Interactive web dashboard for visualizing model and predictions.            |

---

## 🚀 Quick Start

| Step | Command |
|------|---------|
| ✅ Clone Repo | `git clone https://github.com/username/fraud_detection_project.git` |
| 📁 Navigate | `cd fraud_detection_project` |
| 🔧 Create venv | `python -m venv venv` |
| ⚙️ Activate venv | `source venv/bin/activate` *(Windows: `venv\Scripts\activate`)* |
| 📦 Install Deps | `pip install -r requirements.txt` |
| ▶️ Launch App | `streamlit run app.py` |

---

## 📂 Directory Structure

```
fraud_detection_project/
├── app.py                   # Streamlit app for UI
├── creditcard.csv           # Kaggle dataset file
├── data_preprocessing.py    # Preprocessing and SMOTE logic
├── model_training.py        # RandomForest training script
├── real_time_simulation.py  # Simulates live transactions
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

---

## 📊 Dataset Information

| Attribute | Value |
|----------|--------|
| Source | [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) |
| Samples | 284,807 transactions |
| Features | 30 input features (`V1` to `V28`, `Amount`, `Time`) |
| Target | `Class` → 1: Fraud, 0: Legitimate |

> 📁 **Note**: Place `creditcard.csv` in the root directory before running the app.

---

## ⚙️ Workflow Overview


flowchart TD
    A[Load Dataset] --> B[Preprocess Data (Scaling + SMOTE)]
    B --> C[Train RandomForest Model]
    C --> D[Start Real-Time Transaction Simulation]
    D --> E[Live Predictions on Streamlit Dashboard]
```

---

## 🧪 Sample Output

### ✅ Model Training Output

```
Model Accuracy: 0.9846

Classification Report:
              precision    recall  f1-score   support
           0       1.00      0.97      0.98      56962
           1       0.10      0.91      0.18        100

    accuracy                           0.97      57062
   macro avg       0.55      0.94      0.58      57062
weighted avg       1.00      0.97      0.98      57062
```

### 🔁 Real-Time Simulation Output (Sample)

| Transaction ID | Prediction   |
|----------------|--------------|
| 1              | Legitimate   |
| 2              | Legitimate   |
| 3              | Fraudulent ⚠️ |
| 4              | Legitimate   |
| 5              | Fraudulent ⚠️ |

---

## 🌐 Streamlit Dashboard Preview

> **Sections:**
> - Dataset overview table
> - Model training status
> - Accuracy and classification report
> - Button to start real-time fraud simulation
> - Live fraud prediction display

---

## 🧠 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Core language |
| **pandas, numpy** | Data handling |
| **scikit-learn** | ML modeling |
| **imbalanced-learn** | SMOTE for class balancing |
| **Streamlit** | Interactive dashboard |
| **matplotlib / seaborn** *(optional)* | Visualizations |

---

## 🔮 Future Enhancements

| Idea | Description |
|------|-------------|
| 📡 Alerts | Integrate real-time notifications for fraud |
| ☁️ Cloud Deployment | Deploy to AWS/GCP/Streamlit Cloud |
| 📈 Advanced Models | Add LSTM, XGBoost, Autoencoders |
| 📊 Visuals | Enhance dashboard with charts & timelines |

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Libraries: `scikit-learn`, `pandas`, `Streamlit`, `imbalanced-learn`
