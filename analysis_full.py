import os
import ast
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# ============================================================
# LOAD & MERGE
# ============================================================

def load_and_merge(dataset_name):

    unsup_path = f"results/{dataset_name}_unsupervised_full.csv"
    sup_path = f"results/{dataset_name}_supervised_full.csv"

    unsup = pd.read_csv(unsup_path)
    sup = pd.read_csv(sup_path)

    # Normalize supervised schema
    sup["embedding"] = "logistic_regression"
    sup["clustering"] = "supervised"

    combined = pd.concat([unsup, sup], ignore_index=True)
    return combined


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def compute_summary_stats(df, output_dir):

    summary = (
        df.groupby(["embedding", "clustering", "k"])["f1"]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)
    return summary


# ============================================================
# DEGRADATION CURVES
# ============================================================

def plot_degradation(summary, output_dir):

    # KMeans degradation
    plt.figure(figsize=(8,6))
    for emb in summary["embedding"].unique():
        subset = summary[
            (summary["embedding"] == emb) &
            (summary["clustering"] == "kmeans")
        ]
        if not subset.empty:
            plt.plot(subset["k"], subset["mean"], label=f"{emb}")

    plt.xlabel("Number of Classes (k)")
    plt.ylabel("Mean Macro F1")
    plt.title("Degradation Curve - KMeans")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "degradation_kmeans.png"))
    plt.close()

    # Supervised vs Unsupervised
    plt.figure(figsize=(8,6))

    for emb in summary["embedding"].unique():
        subset = summary[
            (summary["embedding"] == emb) &
            (summary["clustering"] == "kmeans")
        ]
        if not subset.empty:
            plt.plot(subset["k"], subset["mean"], label=f"{emb}")

    sup_subset = summary[
        summary["embedding"] == "logistic_regression"
    ]

    if not sup_subset.empty:
        plt.plot(
            sup_subset["k"],
            sup_subset["mean"],
            linestyle="--",
            linewidth=2,
            label="Logistic Regression"
        )

    plt.xlabel("Number of Classes (k)")
    plt.ylabel("Mean Macro F1")
    plt.title("Unsupervised vs Supervised")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "degradation_comparison.png"))
    plt.close()


# ============================================================
# DEGRADATION SLOPES
# ============================================================

def compute_slopes(summary, output_dir):

    slope_records = []

    for emb in summary["embedding"].unique():
        subset = summary[summary["embedding"] == emb]

        X = subset["k"].values.reshape(-1,1)
        y = subset["mean"].values

        model = LinearRegression().fit(X,y)
        slope = model.coef_[0]
        r2 = model.score(X,y)

        slope_records.append({
            "method": emb,
            "slope": slope,
            "r2": r2
        })

    slopes_df = pd.DataFrame(slope_records)
    slopes_df.to_csv(os.path.join(output_dir, "degradation_slopes.csv"), index=False)
    return slopes_df


# ============================================================
# PERFORMANCE GAP
# ============================================================

def compute_gap(summary, output_dir):

    k_values = sorted(summary["k"].unique())
    gap_records = []

    for k in k_values:

        sup_val = summary[
            (summary["embedding"] == "logistic_regression") &
            (summary["k"] == k)
        ]["mean"].values

        if len(sup_val) == 0:
            continue

        sup_val = sup_val[0]

        unsup_methods = summary[
            (summary["embedding"] != "logistic_regression") &
            (summary["k"] == k)
        ]

        for _, row in unsup_methods.iterrows():
            gap_records.append({
                "k": k,
                "unsupervised_method": row["embedding"],
                "gap": sup_val - row["mean"]
            })

    gap_df = pd.DataFrame(gap_records)
    gap_df.to_csv(os.path.join(output_dir, "performance_gap.csv"), index=False)
    return gap_df


# ============================================================
# ROBUSTNESS RATIO
# ============================================================

def compute_robustness(summary, output_dir):

    records = []

    for emb in summary["embedding"].unique():

        f2 = summary[
            (summary["embedding"] == emb) &
            (summary["k"] == 2)
        ]["mean"].values

        fmax = summary[
            (summary["embedding"] == emb) &
            (summary["k"] == summary["k"].max())
        ]["mean"].values

        if len(f2) and len(fmax):
            ratio = fmax[0] / f2[0]
            records.append({
                "method": emb,
                "robustness_ratio": ratio
            })

    rob_df = pd.DataFrame(records)
    rob_df.to_csv(os.path.join(output_dir, "robustness_ratio.csv"), index=False)
    return rob_df


# ============================================================
# CLUSTERING COMPARISON
# ============================================================

def compare_clustering(summary, output_dir):

    clustering_records = []

    for emb in summary["embedding"].unique():

        kmeans = summary[
            (summary["embedding"] == emb) &
            (summary["clustering"] == "kmeans")
        ]

        hac = summary[
            (summary["embedding"] == emb) &
            (summary["clustering"] == "hac")
        ]

        if not kmeans.empty and not hac.empty:

            merged = pd.merge(
                kmeans,
                hac,
                on="k",
                suffixes=("_kmeans", "_hac")
            )

            for _, row in merged.iterrows():
                clustering_records.append({
                    "embedding": emb,
                    "k": row["k"],
                    "difference": row["mean_kmeans"] - row["mean_hac"]
                })

    clustering_df = pd.DataFrame(clustering_records)
    clustering_df.to_csv(os.path.join(output_dir, "clustering_comparison.csv"), index=False)
    return clustering_df


# ============================================================
# BINARY CLASS DIFFICULTY
# ============================================================

def compute_binary_class_difficulty(df, output_dir):

    binary_df = df[df["k"] == 2]

    records = []

    for method in binary_df["embedding"].unique():

        method_df = binary_df[binary_df["embedding"] == method]

        class_scores = {}

        for _, row in method_df.iterrows():
            classes = ast.literal_eval(row["class_subset"])
            f1 = row["f1"]

            for c in classes:
                class_scores.setdefault(c, []).append(f1)

        for c, scores in class_scores.items():
            records.append({
                "method": method,
                "class": c,
                "avg_binary_f1": np.mean(scores)
            })

    difficulty_df = pd.DataFrame(records)
    difficulty_df.to_csv(os.path.join(output_dir, "binary_class_difficulty.csv"), index=False)
    return difficulty_df


# ============================================================
# RANKING PER K
# ============================================================

def compute_ranking(summary, output_dir):

    ranking_records = []

    for k in sorted(summary["k"].unique()):

        subset = summary[summary["k"] == k]
        ranked = subset.sort_values("mean", ascending=False)

        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            ranking_records.append({
                "k": k,
                "method": row["embedding"],
                "rank": rank,
                "mean_f1": row["mean"]
            })

    ranking_df = pd.DataFrame(ranking_records)
    ranking_df.to_csv(os.path.join(output_dir, "ranking_per_k.csv"), index=False)
    return ranking_df


# ============================================================
# MAIN
# ============================================================

def main(dataset_name):

    output_dir = f"results/analysis/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading and merging data...")
    df = load_and_merge(dataset_name)

    print("Computing summary statistics...")
    summary = compute_summary_stats(df, output_dir)

    print("Plotting degradation curves...")
    plot_degradation(summary, output_dir)

    print("Computing degradation slopes...")
    compute_slopes(summary, output_dir)

    print("Computing performance gap...")
    compute_gap(summary, output_dir)

    print("Computing robustness ratio...")
    compute_robustness(summary, output_dir)

    print("Comparing clustering methods...")
    compare_clustering(summary, output_dir)

    print("Computing binary class difficulty...")
    compute_binary_class_difficulty(df, output_dir)

    print("Computing ranking per k...")
    compute_ranking(summary, output_dir)

    print("\nAnalysis complete. Results saved in:", output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    main(args.dataset)

# For Promise dataset:
# python analysis_full.py --dataset promise

# For CrowdRE dataset:
# python analysis_full.py --dataset crowdre