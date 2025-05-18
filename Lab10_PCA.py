import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Step 1: Load data
df = pd.read_csv("/content/drive/MyDrive/MLlab dataset/heart.csv")

# Step 2: Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Step 3: Encode categorical columns using Label Encoding (for binary) or OneHotEncoding (for >2 categories)
df_encoded = df.copy()
label_enc = LabelEncoder()

for col in categorical_cols:
    if df_encoded[col].nunique() == 2:
        df_encoded[col] = label_enc.fit_transform(df_encoded[col])
    else:
        df_encoded = pd.get_dummies(df_encoded, columns=[col])

# Step 4: Separate features and target
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

# Step 5: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train and evaluate models
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

print("=== Accuracy Without PCA ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}: {accuracy_score(y_test, y_pred):.4f}")

# Step 8: Apply PCA to reduce dimensionality
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)

# Step 9: Split PCA-transformed data
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

print("\n=== Accuracy With PCA ===")
for name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    print(f"{name}: {accuracy_score(y_test, y_pred):.4f}")
