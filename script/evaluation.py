import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Load the preprocessed data
data_path = '../data/processed_calories.csv'  # Adjust if necessary
df = pd.read_csv(data_path)

# Load the trained model
model_path = '../data/calorie_burn_model.pkl'
model = joblib.load(model_path)

# Split the data into features and target
X = df.drop(columns=['Calories'])
y = df['Calories']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Actual vs Predicted Calories')
plt.show()
