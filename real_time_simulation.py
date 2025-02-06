import time

def simulate_transactions(model, scaler, data, delay=1):
    for index, row in data.iterrows():
        time.sleep(delay)  # Simulate delay
        transaction = scaler.transform([row.values])
        prediction = model.predict(transaction)
        is_fraud = "Fraudulent" if prediction[0] == 1 else "Legitimate"
        yield index, is_fraud
