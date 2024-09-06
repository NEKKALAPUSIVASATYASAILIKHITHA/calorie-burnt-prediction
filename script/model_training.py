import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the preprocessed data
data_path = '../data/processed_calories.csv'  # Path to preprocessed data
df = pd.read_csv(data_path)

# Assuming 'Calories' is the target variable and 'User_ID' is not a feature
X = df.drop(columns=['Calories'])
y = df['Calories']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Save the trained model
model_path = '../data/calorie_burn_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
