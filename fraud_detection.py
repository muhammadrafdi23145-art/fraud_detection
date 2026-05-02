# ==================================================
# Classification Models: Credit Card Fraud Detection
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. Load Dataset
# Ensure creditcard.csv is in the same directory
df = pd.read_csv("creditcard.csv")

# 2. Preprocessing
# Remove duplicates
df = df.drop_duplicates()

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split data (Stratified to maintain class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Handling Imbalanced Data with SMOTE
print("Class distribution before SMOTE:", np.bincount(y_train))
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", np.bincount(y_train_res))

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training: Logistic Regression
print("\n--- Training Logistic Regression ---")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train_res)
y_pred_log = log_model.predict(X_test_scaled)

print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_log))

# 6. Model Training: Random Forest
print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_res, y_train_res) # Random Forest usually handles raw values well
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf))

# 7. Visualization: Confusion Matrix comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Reds', ax=ax[0])
ax[0].set_title("Confusion Matrix: Logistic Regression")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=ax[1])
ax[1].set_title("Confusion Matrix: Random Forest")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix_comparison.png")
plt.show()

# 8. Export Model
joblib.dump(rf_model, "fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel and Scaler saved successfully!")
