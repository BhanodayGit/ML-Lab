import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ==============================
# Step 1: Load the Dataset
# ==============================
data = pd.read_csv('heart.csv')
print("Dataset loaded successfully!")
print(data.head())

# Convert categorical target column to numeric if necessary
if data['HeartDisease'].dtype == 'object':
    data['HeartDisease'] = data['HeartDisease'].map({'No': 0, 'Yes': 1})

# For this dataset, the target attribute is 'HeartDisease'
target_attr = 'HeartDisease'

# ==============================
# Step 2: Define Functions to Compute Entropy and Information Gain
# ==============================
def entropy(df, target_attr):
    values = df[target_attr].unique()
    entropy_value = 0
    for val in values:
        p = len(df[df[target_attr] == val]) / len(df)
        if p > 0:
            entropy_value -= p * math.log2(p)
    return entropy_value

def information_gain(df, attr, target_attr):
    total_entropy = entropy(df, target_attr)
    values = df[attr].unique()
    weighted_entropy = 0
    for val in values:
        subset = df[df[attr] == val]
        weight = len(subset) / len(df)
        subset_entropy = entropy(subset, target_attr)
        weighted_entropy += weight * subset_entropy
    gain = total_entropy - weighted_entropy
    return gain

# ==============================
# Step 3: Calculate Information Gain for All Features (except target)
# ==============================
attributes = [col for col in data.columns if col != target_attr]
gains = {}
print("\nCalculating Information Gain for each attribute:")
for attr in attributes:
    gain = information_gain(data, attr, target_attr)
    gains[attr] = gain
    print(f"Information Gain for {attr}: {gain:.4f}")

# Determine the root node as the attribute with the highest information gain.
root_node = max(gains, key=gains.get)
print("\nRoot Node selected based on highest information gain:")
print(root_node)

# ==============================
# Step 4: Build a Decision Tree using scikit-learn
# ==============================
# Using the attribute with the highest information gain as the selected feature.
selected_feature = root_node

# Ensure the target variable for sklearn is the same as our target_attr.
X = data[[selected_feature]]
y = data[target_attr]

# Create and train a decision tree with maximum depth 1 (i.e. one split)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(8, 6))
plot_tree(clf, feature_names=[selected_feature], class_names=['No', 'Yes'],
          filled=True, rounded=True)
plt.title("Decision Tree (Depth = 1)")
plt.show()
