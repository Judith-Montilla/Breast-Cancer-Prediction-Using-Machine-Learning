# Breast Cancer Diagnosis Using Machine Learning

# Objective:
# Develop machine learning models to predict breast cancer diagnosis using diagnostic features from imaging data.
# The focus is on accurately classifying tumors as benign or malignant to assist healthcare providers in making informed decisions and improving patient outcomes.

# Key Points:
# - End-to-end analysis covering data preprocessing, model development, and evaluation.
# - Dataset: Kaggle - Breast Cancer Wisconsin (Diagnostic) Dataset.
# - Techniques: Logistic Regression, Random Forest, and XGBoost.
# - Performance Evaluation: Logistic Regression and Random Forest both achieved ROC-AUC scores of 1.00, while XGBoost achieved 0.99.
# - Insights: Features such as radius, texture, and area were identified as the most significant predictors for breast cancer diagnosis.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import shap
import os

# Set the path to save the images
save_path = r'C:\Users\JUDIT\Desktop\Data Sets'

# 1. Data Loading and Overview
# Load the dataset containing features derived from breast cancer images, along with diagnosis labels (M for Malignant, B for Benign)
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
# Drop unnecessary columns that do not contribute to the model's performance
data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# Map 'diagnosis' column to binary labels (Malignant = 1, Benign = 0)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Splitting features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Identifying numeric features for scaling
numeric_features = X.columns.tolist()

# Creating a pipeline for numeric features: scaling
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Applying the numeric transformer
preprocessor = Pipeline(steps=[('num', numeric_transformer)])

# 3. Data Splitting
# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Development: Logistic Regression, Random Forest, and XGBoost with Cross-Validation and Hyperparameter Tuning
# Implementing k-fold cross-validation to improve model reliability
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define models to be used in the pipeline
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', n_jobs=-1)
}

# Function to train, evaluate, and visualize model performance
def train_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, kf):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # Defining parameter grids for each model
    param_grid = {}
    if model_name == 'Logistic Regression':
        param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100]}  # Range of regularization strengths
    elif model_name == 'Random Forest':
        param_grid = {'classifier__n_estimators': [50, 100, 200],  # Number of trees in the forest
                      'classifier__max_depth': [None, 10, 20, 30]}  # Depth of each tree
    elif model_name == 'XGBoost':
        param_grid = {'classifier__learning_rate': [0.01, 0.1, 0.3],  # Learning rate for boosting
                      'classifier__n_estimators': [50, 100, 200]}  # Number of boosting rounds

    # Implementing GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")

    # Making predictions on test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Generating classification report
    print(f"\nClassification Report for {model_name}:\n", classification_report(y_test, y_pred))

    # Calculating and displaying ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"{model_name} ROC-AUC Score: {roc_auc:.2f}")

    # Plotting ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, f'roc_curve_{model_name}.png'))
    plt.show()

    # Plotting Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'PR Curve - {model_name} (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_path, f'precision_recall_curve_{model_name}.png'))
    plt.show()

    # Displaying the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot()
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(save_path, f'confusion_matrix_{model_name}.png'))
    plt.show()

    return best_model

# Loop through models to train, evaluate, and visualize performance
for model_name, model in models.items():
    best_model = train_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, kf)

    # Feature Importance for Logistic Regression
    if model_name == 'Logistic Regression':
        coef = best_model.named_steps['classifier'].coef_[0]
        importance = pd.Series(coef, index=numeric_features)
        
        # Adjust the plot to make sure labels fit
        plt.figure(figsize=(8, 6))
        importance.nlargest(10).plot(kind='barh')
        plt.title(f'Top 10 Important Features - {model_name}')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Features')
        
        # Adjust the layout to ensure all labels are visible
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_path, f'feature_importance_{model_name}.png'))
        plt.show()

    # SHAP values for model interpretability
    if model_name == 'XGBoost':
        explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
        shap_values = explainer.shap_values(X_test)
        
        # SHAP values are used for model interpretability, ensuring transparency in model decision-making.
        shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
        plt.savefig(os.path.join(save_path, f'shap_summary_{model_name}.png'))
        plt.show()

# Conclusion:
# All three models—Logistic Regression, Random Forest, and XGBoost—demonstrated strong performance in classifying breast cancer as benign or malignant.
# Logistic Regression and Random Forest achieved ROC-AUC scores of 1.00, while XGBoost achieved 0.99.
# The models offer healthcare professionals a reliable method for predicting breast cancer, aiding in early diagnosis and treatment.

# Future Work:
# Further exploration could include adding more data, such as genetic information, or integrating the models into real-time clinical workflows.
# Exploring ensemble methods that combine the strengths of these models
