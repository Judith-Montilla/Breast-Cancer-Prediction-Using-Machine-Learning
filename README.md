# Breast Cancer Prediction Using Machine Learning

This repository contains Python code for predicting breast cancer diagnosis using machine learning models. It demonstrates end-to-end data analytics and modeling skills, covering data preprocessing, feature engineering, model development, evaluation, and interpretability.

## Objective

Develop a machine learning model to predict breast cancer diagnosis based on diagnostic imaging features. The focus is on improving diagnostic accuracy to assist healthcare providers in making informed decisions, reducing false negatives, and optimizing clinical workflows.

## Dataset Description

- **Source:** Kaggle - [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- **Features:**
  - **Patient Metrics:** Radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension.
  - **Target Variable:** Diagnosis (Malignant = 1, Benign = 0)

## Methodology Overview

### Data Cleaning and Preprocessing:
- Removed unnecessary columns (e.g., `id`, `Unnamed: 32`) to focus on diagnostic features.
- Converted categorical labels ('diagnosis') to numeric format (Malignant = 1, Benign = 0).
- Standardized numerical features using `StandardScaler` to improve model performance.
- Split the dataset into 80% training and 20% testing sets.

### Exploratory Data Analysis:
- Analyzed feature distributions and identified correlations between features and diagnosis labels.
- Visualized the skewness in diagnostic features to understand their impact on malignancy prediction.

### Model Development:
- Built and evaluated three models: Logistic Regression, Random Forest, and XGBoost.
- Used cross-validation and grid search to fine-tune hyperparameters.
- Metrics used for evaluation include ROC-AUC, Precision, Recall, and F1-Score.

## Model Performance

| Model              | ROC-AUC  | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 1.00     | 0.97      | 0.98   | 0.97     |
| Random Forest       | 1.00     | 0.96      | 0.98   | 0.97     |
| XGBoost             | 0.99     | 0.96      | 0.95   | 0.95     |

## Significant Predictors:
- **Radius**: One of the most important features for identifying malignancy.
- **Texture**: Plays a critical role in distinguishing between benign and malignant tumors.
- **Perimeter**: A key factor in assessing tumor characteristics.

## Visualizations

1. **ROC Curves**: ROC curves illustrate the model performance in distinguishing between benign and malignant tumors.
2. **Precision-Recall Curves**: These curves highlight the trade-off between precision and recall, critical in reducing false positives in cancer diagnosis.
3. **Feature Importance Plot**: Displays the top features influencing model predictions, including radius and perimeter.
4. **SHAP Summary Plot**: Provides model interpretability by showing how individual features contributed to predictions in XGBoost.

## Business Impact

The models demonstrate robust predictive power in diagnosing breast cancer, which can help healthcare providers make more accurate decisions, leading to earlier interventions and more personalized treatment plans. Reducing false negatives can significantly impact patient outcomes and decrease cancer-related mortality rates. Implementing these models in clinical workflows can improve diagnostic accuracy and optimize resource allocation, potentially lowering healthcare costs.

### Examples of Business Impact:
- **Improved Diagnostic Accuracy**: Early identification of malignant tumors can lead to earlier treatments and better patient outcomes.
- **Resource Allocation**: Predictive models can guide hospitals in better allocating resources like medical personnel and diagnostic equipment.
- **Preventive Care**: Identifying high-risk patients early allows for preventive care, improving survival rates and reducing treatment costs.

## Future Work Recommendations

- **Model Improvement**: Investigate the use of advanced ensemble techniques, such as stacking, to improve model accuracy.
- **Real-Time Predictive Tools**: Develop real-time diagnostic tools to assist clinicians at the point of care.
- **Enhanced Data Integration**: Incorporate additional patient data, such as genetic or lifestyle factors, to refine predictions and personalize care.

## Ethical Considerations

- **Data Privacy**: The dataset used does not include personal identifiable information (PII) and complies with privacy regulations like HIPAA.
- **Fairness**: Regular checks were performed to ensure that model predictions are unbiased and fair, ensuring equitable healthcare across diverse populations.
