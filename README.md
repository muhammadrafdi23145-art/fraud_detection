# Credit Card Fraud Detection

**Anomaly Detection | Imbalanced Classification | Financial AI**

## Project Overview
The goal of this project is to detect fraudulent credit card transactions. Fraud detection is a classic **imbalanced classification** problem, as fraudulent transactions are extremely rare compared to legitimate ones.

## Tech Stack
* **Language:** Python
* **Library:** Scikit-Learn, Imbalanced-Learn (SMOTE), Pandas, Seaborn.
* **Algorithms:** Logistic Regression, Random Forest.

## Key Challenges: Handling Data Imbalance
Using raw data results in a biased model that ignores fraud cases. This project implements:
1. **SMOTE:** To synthetically generate fraud samples for better training.
2. **Standard Scaling:** To normalize transaction amounts and time features.
3. **Precision-Recall Optimization:** Focusing on minimizing "False Negatives" (missed fraud).

## Performance
The **Random Forest** model with SMOTE provided the best balance between catching fraud (Recall) and maintaining accuracy.

## Files
- `fraud_detection.py`: Main training pipeline.
- `requirements.txt`: Dependencies.
- `fraud_detection_model.pkl`: Trained model for production.

## Dataset
https://www.kaggle.com/code/aarthiramalingam/creditcard/input

## Result
<img width="1790" height="490" alt="image" src="https://github.com/user-attachments/assets/e71a67cf-bc8e-47f4-aedb-7fc72afd41fb" />

## Evaluation
--- Logistic Regression ---

              precision    recall  f1-score   support

           0       1.00      0.99      1.00     56651
           1       0.14      0.85      0.24        95

    accuracy                           0.99     56746
    macro avg       0.57      0.92      0.62     56746
    weighted avg       1.00      0.99      0.99     56746


--- Training Random Forest ---

              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56651
           1       0.60      0.81      0.69        95

    accuracy                           1.00     56746
    macro avg       0.80      0.90      0.84     56746
    weighted avg       1.00      1.00      1.00     56746

