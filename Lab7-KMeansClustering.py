from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target  # Actual labels (not used in clustering)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Print cluster centers and inertia
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Inertia:", kmeans.inertia_)
print("Silhouette Score:", silhouette_score(X, labels))

# Visualize using first two features: sepal length and width
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label='Centers')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-Means Clustering on Iris (First Two Features)')
plt.legend()
plt.grid(True)
plt.show()
