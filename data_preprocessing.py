import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Data Loaded: {df.head()}")
    return df


def preprocess_data(df):
    # Replace hyphens with NaN for all columns before type conversion
    df = df.replace('-', np.nan)  

    # Convert all columns to numeric, handling errors
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            # Handle columns that cannot be converted to numeric (e.g., categorical)
            pass  

    # Drop rows with NaN in the 'Class' column before splitting 
    df = df.dropna(subset=['Class'])  

    X = df.drop('Class', axis=1)
    y = df['Class']

    # Impute missing values before scaling
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')  # You can choose other strategies
    X = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    print(f"Resampled Data Shape: {X_resampled.shape}, {y_resampled.shape}")
    
    return X_resampled, y_resampled, scaler

def train_model(X_resampled, y_resampled):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # RandomForest with reduced number of estimators and n_jobs=1 (for faster execution)
    model = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=42)  # Reduced number of trees
    
    model.fit(X_train, y_train)  # Fit model
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    return model, accuracy, report, X_test

# Example Usage
file_path = "creditcard.csv" 
df = load_data(file_path)
X_resampled, y_resampled, scaler = preprocess_data(df)
model, accuracy, report, X_test = train_model(X_resampled, y_resampled)