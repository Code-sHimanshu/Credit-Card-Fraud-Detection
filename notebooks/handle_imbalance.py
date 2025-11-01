# Step 2 — Handle Class Imbalance for Credit Card Fraud Detection

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE   # for oversampling
# from imblearn.under_sampling import RandomUnderSampler   # optional alternative

# ======================
# 1. Load dataset
# ======================
PATH = "data/creditcard.csv"   # adjust if needed
df = pd.read_csv(PATH)

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nClass distribution before split:")
print(df["Class"].value_counts(normalize=True))

# ======================
# 2. Split features and target
# ======================
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n✅ Split completed!")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Class distribution in training set:")
print(y_train.value_counts(normalize=True))

# ======================
# 3. Apply SMOTE oversampling
# ======================
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\n✅ After SMOTE:")
print(y_train_res.value_counts())
print("Resampled training shape:", X_train_res.shape)

# ======================
# (Optional) Alternative: Undersampling
# Uncomment below if you prefer random undersampling instead of SMOTE
"""
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
print("\n✅ After undersampling:")
print(y_train_res.value_counts())
print("Resampled training shape:", X_train_res.shape)
"""

# ======================
# 4. Save processed data (optional)
# ======================
X_train_res.to_csv("data/X_train_res.csv", index=False)
y_train_res.to_csv("data/y_train_res.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\n💾 Resampled and test data saved in 'data/' folder!")
print("Step 2 completed successfully ✅")
