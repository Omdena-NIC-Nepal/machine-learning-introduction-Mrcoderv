import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# Define relative paths for GitHub
base_dir = os.path.dirname(__file__)  # Get the directory of the script
data_dir = os.path.join(base_dir, '../data')
model_dir = os.path.join(base_dir, '../models')

# Load test data
X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))

# Load trained model
model = joblib.load(os.path.join(model_dir, 'linear_regression_model.pkl'))

# Predict
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Generate Residual Plot
plt.figure(figsize=(10, 6))
residuals = y_test.values.ravel() - y_pred
plt.scatter(y_pred, residuals, alpha=0.7, color='blue', edgecolor='k', s=50)
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.grid(alpha=0.3)
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
