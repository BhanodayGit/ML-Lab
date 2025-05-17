import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 2: Get the Data
file_path = "/content/heart-attack-risk-prediction-dataset.csv"
df = pd.read_csv(file_path)

# Step 3: Discover and Visualize Data
print(df.info())
print(df.describe())

# Visualize correlations
# Convert 'Gender' to numeric before calculating correlation for the heatmap
df['Gender'] = pd.factorize(df['Gender'])[0]

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 4: Prepare the Data
df = df.drop(columns=["Heart Attack Risk (Text)"])  # Removing redundant target column

# Handling missing values
df.dropna(inplace=True)

# Splitting data
# Keep 'Gender' in X for the ColumnTransformer
X = df.drop(columns=["Heart Attack Risk (Binary)"])
y = df["Heart Attack Risk (Binary)"]

# Preprocessing Pipelines
num_features = X.columns[X.columns != 'Gender'] # Exclude 'Gender' from numerical features
cat_features = ["Gender"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder())
])

# Define the ColumnTransformer with remainder='passthrough'
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features),
    ],
    remainder='passthrough'  # Pass 'Gender' through without transformation
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform features
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Step 5: Select and Train a Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Fine-tune Model
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Step 7: Present Solution
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 8: Launch, Monitor, Maintain
pickle.dump(best_model, open("heart_attack_model.pkl", "wb"))
print("Model saved as heart_attack_model.pkl")
