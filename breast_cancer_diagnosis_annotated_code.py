# Healthcare Cost Prediction Using Regression Analysis

# Objective:
# Develop a regression model to predict healthcare costs using patient demographics, health metrics, and lifestyle factors. 
# The focus is on identifying key cost drivers that help optimize pricing and resource allocation for healthcare providers and insurers.

# Key Points:
# - End-to-end analysis covering data preprocessing, model development, and evaluation.
# - Dataset: Kaggle - Medical Cost Personal Dataset
# - Techniques: Linear regression, feature engineering, and performance evaluation (R²: 0.784, MSE: 33,596,915).
# - Insights: Smoking status, BMI, and age are significant predictors of healthcare costs.

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# 1. Data Loading and Overview
# Load the dataset containing patient demographics, lifestyle factors, and healthcare charges
file_path = r"C:\Users\YourUser\Desktop\Data Sets\insurance.csv"  # Update the path as needed
df = pd.read_csv(file_path)

# Initial data overview: Understanding the structure and summary statistics of the dataset
print(df.head())  # Display the first few rows of the dataset
print(df.describe())  # Summary statistics for numerical features
print(df.info())  # Information about data types and missing values

# 2. Data Preprocessing
# Convert 'sex' to numeric: 1 for male, 0 for female
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)

# Convert 'smoker' to numeric: 1 for yes, 0 for no
df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)

# Drop the 'region' column as it is not needed for the analysis
df.drop(columns=['region'], inplace=True)

# Separate features and target variable
X = df.drop(columns=['charges'])  # Features: 'age', 'sex', 'bmi', 'children', 'smoker'
y = df['charges']  # Target variable

# Standardize features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Model Development and Evaluation

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MSE': mse,
        'R²': r2
    }
    
    print(f"{name} Results:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-Squared (R²): {r2}")
    print("-" * 30)

# Perform cross-validation
kf = KFold(n_splits=10, random_state=42, shuffle=True)
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')
    cv_results[name] = np.mean(cv_scores)
    print(f"{name} Cross-Validation R²: {np.mean(cv_scores)}")

# 4. Assumption Checking
# Residual Analysis and Normality Check
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
residuals = y_train - y_pred_train

# Plot residuals
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_pred_train, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')

plt.subplot(1, 2, 2)
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.show()

# Conclusion:
# The regression models were evaluated using Mean Squared Error (MSE) and R-Squared (R²) metrics. 
# The Linear Regression model performed comparably to Ridge and Lasso regression, with an R² value of 0.784 and an MSE of 33,596,915 on the test set. 
# Significant predictors of healthcare costs include smoking status, BMI, and age. These insights are valuable for healthcare providers aiming to optimize pricing and resource allocation.

# Future Work:
# - **Model Refinement:** Explore advanced models such as Gradient Boosting Machines or Neural Networks for potentially improved performance.
# - **Feature Engineering:** Investigate additional features or interaction terms that may enhance model accuracy.
# - **Deployment:** Consider deploying the model into a real-world application for ongoing cost prediction and optimization.
# - **Ethical Considerations:** Address potential biases and ensure the model’s predictions are used fairly and ethically in healthcare settings.
