import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Define relative paths
base_dir = os.path.dirname(__file__)  # Get the directory of the script
data_dir = os.path.join(base_dir, '../data')
model_dir = os.path.join(base_dir, '../models')

# Load training data
X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

# Train model
model = LinearRegression()
model.fit(X_train, y_train.values.ravel())

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

# Save trained model
joblib.dump(model, os.path.join(model_dir, 'linear_regression_model.pkl'))

# Display coefficients
coef_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_})
print("Model Coefficients:\n", coef_df)
print("Intercept:", model.intercept_)
print(f"Model saved to {os.path.join(model_dir, 'linear_regression_model.pkl')}")