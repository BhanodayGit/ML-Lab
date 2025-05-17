import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load California housing data
california_housing = fetch_california_housing()
california_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
california_data['MEDV'] = california_housing.target

# Features and target
X = california_data.drop('MEDV', axis=1)
y = california_data['MEDV']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Evaluation
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Manual input for prediction
print("\nEnter the following 8 values to predict house price (in $100,000s):")

features = california_housing.feature_names
user_input = []

for feature in features:
    val = float(input(f"{feature}: "))
    user_input.append(val)

manual_data = pd.DataFrame([user_input], columns=features)
predicted_value = rf_regressor.predict(manual_data)

print(f"\nPredicted Median House Value: ${predicted_value[0]*100000:.2f}")
