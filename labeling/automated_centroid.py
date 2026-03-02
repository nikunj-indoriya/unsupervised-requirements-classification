import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_class_centroids(embeddings, labels, class_subset):
    """
    Compute mean embedding for each true class in subset
    """
    centroids = []

    for class_id in class_subset:
        class_points = embeddings[labels == class_id]
        centroids.append(np.mean(class_points, axis=0))

    return np.array(centroids)


def elimination_label_assignment(cluster_centroids, class_centroids):
    """
    One-to-one assignment via elimination strategy
    """
    similarity_matrix = cosine_similarity(cluster_centroids, class_centroids)

    n = similarity_matrix.shape[0]
    assigned = [-1] * n

    for _ in range(n):
        max_idx = np.unravel_index(
            np.argmax(similarity_matrix),
            similarity_matrix.shape
        )

        cluster_i, class_j = max_idx
        assigned[cluster_i] = class_j

        similarity_matrix[cluster_i, :] = -1
        similarity_matrix[:, class_j] = -1

    return assigned


def map_clusters_to_labels(cluster_labels, assignment):
    """
    Convert cluster IDs to predicted class IDs
    """
    predicted = np.zeros_like(cluster_labels)

    for cluster_id, class_id in enumerate(assignment):
        predicted[cluster_labels == cluster_id] = class_id

    return predicted
