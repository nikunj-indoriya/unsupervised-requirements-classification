import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering


def run_kmeans(embeddings, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(embeddings)
    return labels


def run_hac(embeddings, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(embeddings)
    return labels


def compute_cluster_centroids(embeddings, cluster_labels, n_clusters):
    centroids = []
    for i in range(n_clusters):
        cluster_points = embeddings[cluster_labels == i]
        centroids.append(np.mean(cluster_points, axis=0))
    return np.array(centroids)
