import argparse
import numpy as np
from tqdm import tqdm

from datasets.promise_dataset import PromiseDataset
from datasets.crowdre_dataset import CrowdREDataset
from datasets.secreq_dataset import SecReqDataset

from embeddings.load_all_embeddings import get_embedding
from experiments.combination_generator import generate_all_combinations

from clustering.clustering_engine import (
    run_kmeans,
    run_hac,
    run_spectral,
    run_dbscan,
    run_hdbscan,
    compute_cluster_centroids
)

from labeling.automated_centroid import (
    compute_class_centroids,
    elimination_label_assignment,
    map_clusters_to_labels
)

from evaluation.metrics import compute_macro_metrics
from evaluation.experiment_logger import ExperimentLogger

def load_dataset(dataset_name, path):

    if dataset_name == "promise":
        return PromiseDataset(path).load()

    elif dataset_name == "crowdre":
        return CrowdREDataset(path).load()

    elif dataset_name == "secreq":
        return SecReqDataset(path).load()

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def run(dataset_name, data_path):

    dataset = load_dataset(dataset_name, data_path)

    texts = dataset.get_texts()
    labels = dataset.get_labels()
    class_names = dataset.get_class_names()

    class_indices = list(range(len(class_names)))
    combinations = generate_all_combinations(class_indices)

    embeddings_models = [
        "w2v_self",
        "w2v_pretrained",
        "fasttext_pretrained",
        "glove_pretrained",
        "sbert",
        "sroberta",
        "mpnet",
        "e5",
        "instructor"
    ]

    clustering_methods = [
        "kmeans",
        "hac",
        "spectral",
        "dbscan",
        "hdbscan"
    ]

    output_file = f"results/{dataset_name}_unsupervised_full.csv"
    logger = ExperimentLogger(output_file)

    total_skipped = 0
    total_computed = 0

    for model_name in embeddings_models:

        print(f"\nProcessing embedding: {model_name}")

        embeddings = get_embedding(model_name, texts, dataset_name)

        for combo in tqdm(combinations):

            k = len(combo)
            combo = list(combo)

            mask = np.isin(labels, combo)

            subset_embeddings = embeddings[mask]
            subset_labels = labels[mask]

            label_map = {cls: i for i, cls in enumerate(combo)}
            subset_labels = np.array([label_map[l] for l in subset_labels])

            for clustering_method in clustering_methods:

                if logger.is_completed(k, combo, model_name, clustering_method):
                    total_skipped += 1
                    continue

                if clustering_method == "kmeans":
                    cluster_labels = run_kmeans(subset_embeddings, k)

                elif clustering_method == "hac":
                    cluster_labels = run_hac(subset_embeddings, k)

                elif clustering_method == "spectral":
                    cluster_labels = run_spectral(subset_embeddings, k)

                elif clustering_method == "dbscan":
                    cluster_labels = run_dbscan(subset_embeddings)

                elif clustering_method == "hdbscan":
                    cluster_labels = run_hdbscan(subset_embeddings)

                else:
                    raise ValueError(f"Unknown clustering method: {clustering_method}")

                unique_clusters = set(cluster_labels)

                # remove noise cluster if present
                if -1 in unique_clusters:
                    unique_clusters.remove(-1)

                # skip if cluster count != k
                if len(unique_clusters) != k:
                    continue

                cluster_centroids = compute_cluster_centroids(
                    subset_embeddings,
                    cluster_labels,
                    k
                )

                class_centroids = compute_class_centroids(
                    subset_embeddings,
                    subset_labels,
                    list(range(k))
                )

                assignment = elimination_label_assignment(
                    cluster_centroids,
                    class_centroids
                )

                predicted = map_clusters_to_labels(
                    cluster_labels,
                    assignment
                )

                metrics = compute_macro_metrics(
                    subset_labels,
                    predicted
                )

                logger.log(
                    k,
                    combo,
                    model_name,
                    clustering_method,
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"]
                )

                total_computed += 1

    print("\nExperiment Finished")
    print("Total computed:", total_computed)
    print("Total skipped (resume):", total_skipped)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)

    args = parser.parse_args()

    run(args.dataset, args.path)


# Example runs
# python run_full_unsupervised.py --dataset promise --path data/PROMISE_exp.arff
# python run_full_unsupervised.py --dataset crowdre --path data/requirements.csv