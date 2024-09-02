# Breast-Cancer-Prediction-Using-Machine-Learning

This repository features Python code for predicting breast cancer diagnoses using machine learning models, demonstrating end-to-end data analytics and modeling skills. It covers data preprocessing, feature engineering, model development, and evaluation, offering practical insights into breast cancer diagnosis and treatment.

## Objective

Develop machine learning models to predict breast cancer diagnoses based on patient features and diagnostic data. The focus is on enhancing diagnostic accuracy to support better clinical decision-making and improve patient outcomes.

## Dataset Description

- **Source:** Kaggle - Breast Cancer Wisconsin (Diagnostic) Dataset
- **Features:**
  - **Patient Metrics:** Radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.
  - **Diagnosis:** Malignant or benign (target variable)

## Methodology Overview

- **Data Cleaning and Preprocessing:**
  - Converted categorical labels ('diagnosis') to numeric format (Malignant=1, Benign=0).
  - Handled missing data and standardized numerical features.
  - Split data into training and testing sets.

- **Exploratory Data Analysis:**
  - Analyzed feature distributions, correlations, and relationships with the target variable.
  - Identified significant predictors of breast cancer diagnoses.

- **Modeling:**
  - Built and tuned multiple machine learning models: Logistic Regression, Random Forest, and XGBoost.
  - Evaluated model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
  - Cross-validation employed to ensure robustness and generalizability.

## Key Findings

- **Model Performance:**
  - **Logistic Regression:**
    - **Accuracy:** 97.02%
    - **Precision:** 97.43%
    - **Recall:** 96.57%
    - **F1-Score:** 97.00%
    - **ROC-AUC:** 0.97
  - **Random Forest:**
    - **Accuracy:** 98.01%
    - **Precision:** 98.15%
    - **Recall:** 97.43%
    - **F1-Score:** 97.79%
    - **ROC-AUC:** 0.98
  - **XGBoost:**
    - **Accuracy:** 98.12%
    - **Precision:** 98.26%
    - **Recall:** 97.57%
    - **F1-Score:** 97.91%
    - **ROC-AUC:** 0.98

- **Significant Predictors:** 
  - Radius, texture, perimeter, and area are the top features influencing the diagnosis.

## Business Impact

The models demonstrate high accuracy and robustness in predicting breast cancer diagnoses. By leveraging these models, healthcare providers can enhance diagnostic accuracy and improve patient outcomes. Future work could involve deploying these models in clinical settings and integrating them with diagnostic workflows to further validate their effectiveness and impact.

## Future Work Recommendations

- Implement and test the models in a real-world clinical environment to assess practical impact.
- Explore additional features or data sources to enhance model performance and robustness.
- Investigate the application of advanced machine learning techniques or deep learning models for improved diagnostic accuracy.

## Ethical Considerations

When deploying predictive models in healthcare, it’s crucial to ensure that predictions are used responsibly and do not perpetuate biases or inaccuracies. Regular updates and monitoring of the model are necessary to maintain its accuracy and fairness.
