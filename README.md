# Breast Cancer Diagnosis Using Machine Learning

## Objective
The primary objective of this project is to develop machine learning models to predict breast cancer diagnosis using diagnostic features derived from imaging data. The goal is to classify tumors as benign or malignant with high accuracy, supporting healthcare providers in making informed decisions and improving patient outcomes.

## Dataset Description
- **Source:** Kaggle - [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Features:**
  - Diagnostic features from medical imaging, including mean, standard error, and worst values for various tumor characteristics such as radius, texture, smoothness, and concavity.
  - **Target Variable:** Diagnosis (M = Malignant, B = Benign)

## Methodology Overview

### Data Cleaning and Preprocessing
- Dropped unnecessary columns (`id`, `Unnamed: 32`).
- Mapped categorical target variable `diagnosis` to numeric values (Malignant = 1, Benign = 0).
- Scaled numerical features using **StandardScaler** for uniformity during model training.
- Split the dataset into training and test sets (80/20 split).

### Modeling
- Developed and compared the following models to classify tumors:
  - **Logistic Regression**
  - **Random Forest Classifier**
  - **Support Vector Machine (SVM)**
  - **XGBoost**
- Used **GridSearchCV** for hyperparameter tuning and **cross-validation** to avoid overfitting.
- Evaluated each model using metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC**.

## Model Performance

| Model                 | Test Accuracy | Test ROC-AUC | Best Parameters                       |
|-----------------------|---------------|--------------|---------------------------------------|
| Logistic Regression    | 97%           | 1.00         | {'C': 1}                              |
| Random Forest          | 96%           | 1.00         | {'max_depth': None, 'n_estimators': 100} |
| Support Vector Machine | 96%           | 1.00         | {'C': 1, 'kernel': 'linear'}          |
| XGBoost                | 96%           | 0.99         | {'learning_rate': 0.3, 'n_estimators': 200} |

### Key Visualizations
- **Feature Importance (Random Forest)**: Highlighted the most significant features in predicting tumor diagnosis, such as `concave points_mean`, `radius_worst`, and `area_worst`.
- **ROC Curves**: Visualized the true positive rate vs. false positive rate for each model, demonstrating excellent performance across all models.
- **Confusion Matrices**: Evaluated classification accuracy, showing the number of correct vs. incorrect classifications for each model.
- **SHAP Values (XGBoost)**: Explained the impact of individual features on model predictions, providing insight into the most influential diagnostic features.

## Business Impact
This project demonstrates how machine learning can be applied to healthcare, offering the following benefits:
1. **Early Detection**: High-accuracy models like Logistic Regression and Random Forest can assist doctors in diagnosing breast cancer earlier, improving patient outcomes.
2. **Clinical Decision Support**: These models can be integrated into clinical workflows to provide real-time, data-driven support for medical professionals.
3. **Resource Optimization**: Accurate predictions can help optimize the allocation of medical resources, ensuring high-risk patients are prioritized for further testing or treatment.

## Collaborative Experience
This machine learning approach can serve as a valuable tool for cross-functional healthcare teams, including radiologists, oncologists, and hospital administrators, to make data-driven decisions and improve overall care quality.

## Future Work Recommendations
- **Model Improvement**: Explore more complex models, such as **Gradient Boosting** or **Ensemble Methods**, to further increase predictive power.
- **Real-Time Prediction**: Implement these models in real-time diagnostic tools to assist healthcare providers with immediate, accurate assessments.
- **Data Expansion**: Incorporate additional features, such as genetic data or patient history, to further refine diagnostic predictions.

## Ethical Considerations
- **Data Privacy**: Although the dataset used is publicly available and anonymized, it is important to ensure compliance with **HIPAA** and other regulations when applying models to patient data.
- **Bias Mitigation**: Model performance was carefully evaluated to avoid bias, ensuring fair treatment across different patient demographics.
