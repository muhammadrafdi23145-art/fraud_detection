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
<img width="1175" height="490" alt="image" src="https://github.com/user-attachments/assets/486e454c-b778-4423-8f21-a82be7923ae3" />
LR accuracy 0.99
RF accuracy 1.00

