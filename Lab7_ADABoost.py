import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = pd.read_csv('/content/Iris (1).csv')

# Features and labels
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# ---------------- Model 1: AdaBoost with Decision Tree ----------------
dt_model = AdaBoostClassifier(n_estimators=50, random_state=0)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
y_proba_dt = dt_model.predict_proba(X_test)

# ---------------- Model 2: AdaBoost with SVC ----------------
svc = SVC(probability=True, kernel='linear')
svc_model = AdaBoostClassifier(estimator=svc, n_estimators=50, random_state=0)
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)
y_proba_svc = svc_model.predict_proba(X_test)

# ---------------- Evaluation Function ----------------
def evaluate_model(name, y_test, y_pred, y_proba):
    print(f"\n{name} Evaluation Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC-AUC Score for multiclass
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
    print("ROC-AUC Score (OvR):", roc_auc)

# ---------------- Evaluate both models ----------------
evaluate_model("AdaBoost (Decision Tree)", y_test, y_pred_dt, y_proba_dt)
evaluate_model("AdaBoost (SVC)", y_test, y_pred_svc, y_proba_svc)
