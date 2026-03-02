import argparse
import numpy as np
from tqdm import tqdm

from datasets.promise_dataset import PromiseDataset
from datasets.crowdre_dataset import CrowdREDataset
from datasets.secreq_dataset import SecReqDataset

from experiments.combination_generator import generate_all_combinations
from evaluation.experiment_logger import ExperimentLogger

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support


# ============================================================
# Dataset Loader
# ============================================================

def load_dataset(dataset_name, path):

    if dataset_name == "promise":
        return PromiseDataset(path).load()

    elif dataset_name == "crowdre":
        return CrowdREDataset(path).load()

    elif dataset_name == "secreq":
        return SecReqDataset(path).load()

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


# ============================================================
# Metrics
# ============================================================

def compute_macro_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
    return precision, recall, f1


# ============================================================
# Main Experiment
# ============================================================

def run(dataset_name, data_path):

    dataset = load_dataset(dataset_name, data_path)

    texts = np.array(dataset.get_texts())
    labels = np.array(dataset.get_labels())

    class_indices = list(range(len(dataset.get_class_names())))
    combinations = generate_all_combinations(class_indices)

    output_file = f"results/{dataset_name}_supervised_full.csv"
    logger = ExperimentLogger(output_file)

    total_skipped = 0
    total_computed = 0

    for combo in tqdm(combinations):

        k = len(combo)
        combo = list(combo)

        # Resume check BEFORE doing anything expensive
        if logger.is_completed(k, combo, "logistic_regression", "supervised"):
            total_skipped += 1
            continue

        mask = np.isin(labels, combo)
        subset_texts = texts[mask]
        subset_labels = labels[mask]

        # Remap labels
        label_map = {cls: i for i, cls in enumerate(combo)}
        subset_labels = np.array([label_map[l] for l in subset_labels])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_precisions = []
        fold_recalls = []
        fold_f1s = []

        for train_idx, test_idx in skf.split(subset_texts, subset_labels):

            X_train, X_test = subset_texts[train_idx], subset_texts[test_idx]
            y_train, y_test = subset_labels[train_idx], subset_labels[test_idx]

            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2
            )

            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            param_grid = {
                "C": [0.01, 0.1, 1, 10]
            }

            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs"
            )

            grid = GridSearchCV(
                clf,
                param_grid,
                cv=3,
                scoring="f1_macro",
                n_jobs=-1
            )

            grid.fit(X_train_tfidf, y_train)

            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test_tfidf)

            precision, recall, f1 = compute_macro_metrics(y_test, y_pred)

            fold_precisions.append(precision)
            fold_recalls.append(recall)
            fold_f1s.append(f1)

        # Log using new logger signature
        logger.log(
            k,
            combo,
            "logistic_regression",
            "supervised",
            round(np.mean(fold_precisions), 4),
            round(np.mean(fold_recalls), 4),
            round(np.mean(fold_f1s), 4)
        )

        total_computed += 1

    print("\nSupervised Experiment Finished")
    print("Total computed:", total_computed)
    print("Total skipped (resume):", total_skipped)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)

    args = parser.parse_args()

    run(args.dataset, args.path)

# For promise dataset,
# python run_logistic_exhaustive.py --dataset promise --path data/PROMISE_exp.arff

# For CrowdRE dataset,
# python run_logistic_exhaustive.py --dataset crowdre --path data/requirements.csv