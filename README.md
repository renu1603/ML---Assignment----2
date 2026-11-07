# Machine Learning Classification Benchmark & Streamlit Application

## a. Problem Statement
The objective of this project is to implement, evaluate, and compare five supervised machine learning classification algorithms to predict the presence of heart disease based on medical diagnostic measurements. An interactive web app is deployed via Streamlit Community Cloud to allow dynamic uploading of test datasets and instant performance calculation using pre-trained model artifacts without retraining.

## b. Dataset Description
* **Dataset Source**: UCI Heart Disease Dataset
* **Instance Count**: 1,025 instances
* **Feature Count**: 13 numerical and categorical input features.
* **Target Column**: `target` (0 = No Heart Disease, 1 = Heart Disease Present).

## c. GitHub Repository Link
[Your GitHub Repository Link Here]

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8585 | 0.9254 | 0.8304 | 0.9038 | 0.8655 | 0.7188 |
| **Decision Tree** | 0.9122 | 0.9410 | 0.8879 | 0.9423 | 0.9143 | 0.8258 |
| **KNN** | 0.8683 | 0.9298 | 0.8407 | 0.9135 | 0.8756 | 0.7383 |
| **Naive Bayes** | 0.8244 | 0.8986 | 0.8036 | 0.8654 | 0.8333 | 0.6502 |
| **Support Vector Machine** | 0.9171 | 0.9631 | 0.8889 | 0.9519 | 0.9192 | 0.8358 |

## Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Serves as a reliable linear baseline, achieving 85.85% accuracy and strong recall (90.38%), indicating effective identification of positive cases. |
| **Decision Tree** | Captures non-linear feature interactions effectively, achieving high accuracy (91.22%) and balanced precision-recall without severe overfitting under depth control. |
| **KNN** | Delivers strong local distance-based predictions (86.83% accuracy) following standard feature scaling. |
| **Naive Bayes** | Yields slightly lower accuracy (82.44%) due to strict feature independence assumptions across correlated clinical metrics. |
| **Support Vector Machine** | Outperforms all other models across all 6 metrics, reaching 91.71% accuracy, 0.9631 AUC, and a top MCC score of 0.8358. |
| **Overall Winner** | **Support Vector Machine (SVM)** is the clear winner for this dataset, providing optimal boundary separation with the highest overall AUC and MCC. |
