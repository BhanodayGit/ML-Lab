import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('IRIS.csv')

print("First 5 rows of the dataset:")
print(df.head())

# Feature and target separation
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Model
k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy on test set: {accuracy:.2f}")
print("\nPredictions on test set:")
print(y_pred)

# ----------- Manual Input Section -----------
print("\nEnter values for a new flower (sepal_length, sepal_width, petal_length, petal_width):")
input_values = list(map(float, input().split()))

# Convert to 2D array for prediction
manual_sample = [input_values]
manual_prediction = model.predict(manual_sample)

print(f"Prediction for manual input: {manual_prediction[0]}")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
