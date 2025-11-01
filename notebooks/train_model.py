import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import dump

# ======================
# 1. Load preprocessed data
# ======================
print("📂 Loading resampled and test data...")
X_train_res = pd.read_csv("data/X_train_res.csv")
y_train_res = pd.read_csv("data/y_train_res.csv").squeeze()
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

print("✅ Data loaded successfully!")
print("Train shape:", X_train_res.shape)
print("Test shape:", X_test.shape)

# ======================
# 2. Train Logistic Regression
# ======================
print("\n⚙️ Training Logistic Regression model...")
log_reg = LogisticRegression(max_iter=500, solver='liblinear')
log_reg.fit(X_train_res, y_train_res)

# Save trained model
os.makedirs("models", exist_ok=True)
dump(log_reg, "models/fraud_model.joblib")
print("💾 Model saved at models/fraud_model.joblib")

# Predictions
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

# ======================
# 3. Evaluate Performance
# ======================
print("\n📊 Evaluation Metrics (Logistic Regression):")
report = classification_report(y_test, y_pred, digits=4)
print(report)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Confusion Matrix:\n", cm)
print("ROC-AUC Score:", roc_auc)

# ======================
# 4. Save Confusion Matrix Plot
# ======================
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit (0)", "Fraud (1)"],
            yticklabels=["Legit (0)", "Fraud (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()
print("📸 Confusion matrix saved at results/confusion_matrix.png")

# ======================
# 5. Plot and Save ROC Curve
# ======================
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_curve.png")
plt.close()
print("📈 ROC curve saved at results/roc_curve.png")

# ======================
# 6. Save metrics summary to text file
# ======================
with open("results/metrics_summary.txt", "w") as f:
    f.write("=== Logistic Regression Evaluation ===\n\n")
    f.write(report)
    f.write(f"\nROC-AUC Score: {roc_auc:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
print("📝 Metrics summary saved at results/metrics_summary.txt")

print("\n✅ Step 3 completed successfully!")
