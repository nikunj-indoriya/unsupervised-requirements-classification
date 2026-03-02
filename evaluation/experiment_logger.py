import csv
import os
import ast


class ExperimentLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.completed = set()

        # If file exists → load completed experiments
        if os.path.exists(self.file_path):
            self._load_completed()
        else:
            self._create_file()

    # --------------------------------------------------
    # Create file with header
    # --------------------------------------------------
    def _create_file(self):
        with open(self.file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "k",
                "class_subset",
                "embedding",
                "clustering",
                "precision",
                "recall",
                "f1"
            ])

    # --------------------------------------------------
    # Load existing experiments for resume
    # --------------------------------------------------
    def _load_completed(self):
        with open(self.file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = int(row["k"])

                # Handle "[1, 2]" or "(1, 2)"
                subset = ast.literal_eval(row["class_subset"])
                subset = tuple(sorted(subset))

                embedding = row["embedding"]
                clustering = row["clustering"]

                key = (k, subset, embedding, clustering)
                self.completed.add(key)

    # --------------------------------------------------
    # Check if experiment already done
    # --------------------------------------------------
    def is_completed(self, k, subset, embedding, clustering):
        subset = tuple(sorted(subset))
        key = (k, subset, embedding, clustering)
        return key in self.completed

    # --------------------------------------------------
    # Log new experiment
    # --------------------------------------------------
    def log(self, k, subset, embedding, clustering, precision, recall, f1):

        subset = tuple(sorted(subset))
        key = (k, subset, embedding, clustering)

        if key in self.completed:
            return  # Skip silently

        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                k,
                list(subset),   # Keep same format as before
                embedding,
                clustering,
                precision,
                recall,
                f1
            ])

        self.completed.add(key)