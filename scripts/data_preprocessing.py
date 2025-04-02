import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define relative path for GitHub
data_path = os.path.join(os.path.dirname(__file__), '../data/boston_housing.csv')

# Load dataset
df = pd.read_csv(data_path)

# Data preprocessing
df = df.dropna()

X = df.drop('medv', axis=1)
y = df['medv']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ensure the directory exists
output_dir = os.path.join(os.path.dirname(__file__), '../data')
os.makedirs(output_dir, exist_ok=True)

# Save preprocessed data
X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

# Print dataset shapes
print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
