import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = '../data/calories.csv'  # Updated path based on your dataset name
df = pd.read_csv(data_path)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handling missing values (if any) - let's fill with mean for numeric columns
df.fillna(df.mean(), inplace=True)

# Feature Scaling (if necessary) - scaling numeric features
scaler = StandardScaler()
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save the preprocessed data
df.to_csv('../data/processed_calories.csv', index=False)  # Updated file name
print("\nPreprocessed Data Saved.")
