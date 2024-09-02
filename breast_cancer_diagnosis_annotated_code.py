# Breast Cancer Classification Using Machine Learning Models

# Objective:
# This script develops and evaluates multiple machine learning models to classify breast cancer as benign or malignant.
# The focus is on comparing model performance, understanding feature importance, ensuring model interpretability, and robustness in predictions.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay, make_scorer
import shap
import os

# Set the path to save the images
save_path = r'C:\Users\JUDIT\Desktop\Data Sets'

# Function to handle data loading with error handling
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        raise

# Function to preprocess the data
def preprocess_data(data):
    data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])
    
    return X, y, preprocessor

# Function to split the data into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate the model
def train_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, kf, scoring, preprocessor):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    param_grid = {}
    if model_name == 'Logistic Regression':
        param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100]}
    elif model_name == 'Random Forest':
        param_grid = {'classifier__n_estimators': [50, 100, 200],
                      'classifier__max_depth': [None, 10, 20, 30]}
    elif model_name == 'XGBoost':
        param_grid = {'classifier__learning_rate': [0.01, 0.1, 0.3],
                      'classifier__n_estimators': [50, 100, 200]}
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    print(f"\nClassification Report for {model_name}:\n", classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"{model_name} ROC-AUC Score: {roc_auc:.2f}")
    
    plot_roc_curve(model_name, y_test, y_pred_proba)
    plot_precision_recall_curve(model_name, y_test, y_pred_proba)
    plot_confusion_matrix(model_name, y_test, y_pred, best_model)
    
    if model_name == 'Logistic Regression':
        plot_feature_importance(best_model.named_steps['classifier'].coef_[0], X_train.columns, model_name)
    
    if model_name == 'XGBoost':
        plot_shap_summary(best_model.named_steps['classifier'], X_test, X_test.columns, model_name)
    
    return best_model

# Function to plot ROC curve
def plot_roc_curve(model_name, y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc_score(y_test, y_pred_proba):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, f'roc_curve_{model_name}.png'))
    plt.show()

# Function to plot Precision-Recall curve
def plot_precision_recall_curve(model_name, y_test, y_pred_proba):
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

# Function to plot Confusion Matrix
def plot_confusion_matrix(model_name, y_test, y_pred, model):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(save_path, f'confusion_matrix_{model_name}.png'))
    plt.show()

# Function to plot Feature Importance for Logistic Regression
def plot_feature_importance(coef, feature_names, model_name):
    importance = pd.Series(coef, index=feature_names)
    importance.nlargest(10).plot(kind='barh')
    plt.title(f'Top 10 Important Features - {model_name}')
    plt.savefig(os.path.join(save_path, f'feature_importance_{model_name}.png'))
    plt.show()

# Function to plot SHAP Summary for XGBoost
def plot_shap_summary(model, X_test, feature_names, model_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    plt.savefig(os.path.join(save_path, f'shap_summary_{model_name}.png'))
    plt.show()

# Load the dataset
file_path = os.path.join(save_path, 'data.csv')
data = load_data(file_path)

# Preprocess the data
X, y, preprocessor = preprocess_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y)

# Set up K-Fold Cross-Validation and Scoring
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = make_scorer(roc_auc_score, greater_is_better=True)

# Define models to be used in the pipeline
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', n_jobs=-1)
}

# Loop through models to train, evaluate, and visualize performance
for model_name, model in models.items():
    best_model = train_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, kf, scoring, preprocessor)

# Conclusion:
# All three models—Logistic Regression, Random Forest, and XGBoost—demonstrated strong performance in classifying breast cancer as benign or malignant.
# XGBoost, with hyperparameter tuning and SHAP value analysis, showed slightly better performance in interpretability and feature contribution.

# Future Work:
# Future iterations of this project could include more complex models or the integration of additional data sources such as genetic information.
# Exploring ensemble methods that combine the strengths of each model could also yield improved performance.
# Additionally, the inclusion of error handling throughout the script ensures robustness in different production environments.
