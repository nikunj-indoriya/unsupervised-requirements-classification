import numpy as np

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
    DBSCAN
)

try:
    import hdbscan
except ImportError:
    hdbscan = None

def run_kmeans(embeddings, n_clusters):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(embeddings)
    return labels


def run_hac(embeddings, n_clusters):
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )
    labels = model.fit_predict(embeddings)
    return labels

def run_spectral(embeddings, n_clusters):
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        assign_labels="kmeans",
        random_state=42
    )
    labels = model.fit_predict(embeddings)
    return labels

def run_dbscan(embeddings, eps=0.5, min_samples=5):
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean"
    )
    labels = model.fit_predict(embeddings)
    return labels

def run_hdbscan(embeddings, min_cluster_size=5):

    if hdbscan is None:
        raise ImportError(
            "hdbscan package is not installed. Install with: pip install hdbscan"
        )

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean"
    )

    labels = model.fit_predict(embeddings)
    return labels

def compute_cluster_centroids(embeddings, cluster_labels, n_clusters):

    centroids = []

    for i in range(n_clusters):
        cluster_points = embeddings[cluster_labels == i]

        if len(cluster_points) == 0:
            centroids.append(np.zeros(embeddings.shape[1]))
        else:
            centroids.append(np.mean(cluster_points, axis=0))

    return np.array(centroids)
