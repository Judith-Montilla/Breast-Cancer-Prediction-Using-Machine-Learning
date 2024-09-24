# Breast Cancer Diagnosis Using Machine Learning

# Objective:
# Develop machine learning models to predict breast cancer diagnosis using diagnostic features from imaging data.
# The focus is on accurately classifying tumors as benign or malignant to assist healthcare providers in making informed decisions and improving patient outcomes.

# Key Points:
# - End-to-end analysis covering data preprocessing, model development, and evaluation.
# - Dataset: Kaggle - Breast Cancer Wisconsin (Diagnostic) Dataset.
# - Techniques: Logistic Regression, Random Forest, SVM, and XGBoost.
# - Performance Evaluation: Evaluating models using metrics such as ROC-AUC, Precision, Recall, F1-Score, and Confusion Matrices.
# - Insights: Features such as radius, texture, and area were identified as the most significant predictors for breast cancer diagnosis.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import shap
import os

# Set the path to save the images
save_path = r'C:\Users\JUDIT\Desktop\Data Sets'

# 1. Data Loading and Overview
file_path = os.path.join(save_path, 'data.csv')
try:
    data = pd.read_csv(file_path)
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    raise

# Display basic information about the dataset
print("First 5 rows of the dataset:\n", data.head())
print("Dataset description:\n", data.describe())
print("Missing values in each column:\n", data.isnull().sum())

# 2. Data Preprocessing
# Drop unnecessary columns and handle missing values
data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

numeric_features = X.columns.tolist()

# Scale the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Feature Selection (Recursive Feature Elimination and RandomForest's Feature Importance)
from sklearn.feature_selection import RFE

# Recursive Feature Elimination
selector = RFE(estimator=LogisticRegression(max_iter=10000), n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)
selected_features = [numeric_features[i] for i in range(len(selector.support_)) if selector.support_[i]]
print(f"Top 10 Selected Features: {selected_features}")

# Feature Importance from RandomForest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_

# Plotting feature importance
plt.figure(figsize=(10,6))
pd.Series(importances, index=numeric_features).nlargest(10).plot(kind='barh', color='skyblue')
plt.title('Top 10 Important Features - Random Forest', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig(os.path.join(save_path, 'rf_feature_importance.png'))
plt.show()

# 5. Model Development and Evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True),
    'XGBoost': XGBClassifier(eval_metric='logloss', n_jobs=-1)
}

def train_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, kf):
    param_grid = {}
    if model_name == 'Logistic Regression':
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    elif model_name == 'Random Forest':
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
    elif model_name == 'Support Vector Machine':
        param_grid = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    elif model_name == 'XGBoost':
        param_grid = {'learning_rate': [0.01, 0.1, 0.3], 'n_estimators': [50, 100, 200]}

    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")

    # Predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Classification Report
    print(f"\nClassification Report for {model_name}:\n", classification_report(y_test, y_pred))

    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"{model_name} ROC-AUC Score: {roc_auc:.2f}")

    # ROC Curve - With Grid Lines
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.grid(True)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(os.path.join(save_path, f'roc_curve_{model_name}.png'))
    plt.show()

    # Confusion Matrix - Updated Color Scheme
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.savefig(os.path.join(save_path, f'confusion_matrix_{model_name}.png'))
    plt.show()

    return best_model

# Train and evaluate each model
best_models = {}
for model_name, model in models.items():
    best_models[model_name] = train_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, kf)

# 6. SHAP Values for XGBoost and Random Forest
explainer = shap.TreeExplainer(best_models['XGBoost'])
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test, feature_names=numeric_features)
plt.title('SHAP Summary Plot for XGBoost', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig(os.path.join(save_path, 'shap_summary_xgboost.png'))
plt.show()

# Conclusion:
# - All four models performed well, with ROC-AUC scores close to 1.00.
# - The confusion matrices show high accuracy, with few misclassifications.
# - SHAP values explain the importance of features, making models interpretable and useful for clinical decision-making.
