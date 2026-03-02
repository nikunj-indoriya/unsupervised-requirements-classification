from sklearn.metrics import precision_recall_fscore_support


def compute_macro_metrics(true_labels, predicted_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }
