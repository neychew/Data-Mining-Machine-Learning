"""
Pertemuan 5: Unsupervised Learning - Clustering & Dimensionality Reduction
Topic: PCA (Principal Component Analysis) & K-Means Clustering
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

print("=== PERTEMUAN 5: PCA & K-MEANS CLUSTERING ===")

# Load Data
iris = load_iris()
X = iris.data

# 1. Reduksi Dimensi dengan PCA menjadi 2 Komponen Utama
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 2. K-Means Clustering (3 Cluster)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# 3. Visualisasi Hasil Clustering pada Ruang 2D PCA
plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolors='k')
plt.title("K-Means Clustering via 2D PCA Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster ID')
plt.show()
