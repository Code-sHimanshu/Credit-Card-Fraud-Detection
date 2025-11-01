import pandas as pd
import time
import os
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

# ======================
# 1. Load data
# ======================
print("📂 Loading data...")
X_train_res = pd.read_csv("data/X_train_res.csv")
y_train_res = pd.read_csv("data/y_train_res.csv").squeeze()
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

print("✅ Data loaded successfully!")

# ======================
# 2. Train or load model
# ======================
MODEL_PATH = "models/fraud_model.joblib"
os.makedirs("models", exist_ok=True)

try:
    model = load(MODEL_PATH)
    print(f"✅ Loaded saved model from {MODEL_PATH}")
except:
    print("⚙️ Training new Logistic Regression model...")
    model = LogisticRegression(max_iter=500, solver="liblinear")
    model.fit(X_train_res, y_train_res)
    dump(model, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")


# 3. Real-time simulation

print("\n🚀 Starting real-time fraud detection simulation...")
sample_data = X_test.sample(20, random_state=42)  # simulate 20 incoming transactions

results = []  # store outputs here

for i, row in enumerate(sample_data.itertuples(index=False), 1):
    transaction = [list(row)]
    prob = model.predict_proba(transaction)[0][1]
    prediction = "FRAUD ⚠️" if prob > 0.5 else "Legitimate ✅"

    print(f"Transaction {i}:  Predicted → {prediction} (Fraud probability: {prob:.4f})")

    # save result
    row_dict = sample_data.iloc[i - 1].to_dict()
    row_dict["Fraud_Probability"] = prob
    row_dict["Prediction"] = prediction
    results.append(row_dict)

    time.sleep(0.5)  # simulate delay between transactions


# 4. Save simulation results

os.makedirs("results", exist_ok=True)
results_df = pd.DataFrame(results)
output_path = "results/realtime_predictions.csv"
results_df.to_csv(output_path, index=False)

print(f"\n💾 All 20 transactions saved to: {output_path}")
print("🎯 Real-time simulation completed successfully!")
