import pandas as pd
import numpy as np


class CrowdREDataset:

    def __init__(self, file_path):
        self.file_path = file_path
        self.texts = []
        self.labels = []
        self.class_names = None
        self.label_encoder = None
        self.label_decoder = None

    def load(self):

        df = pd.read_csv(self.file_path)

        # Keep only required columns
        df = df[["feature", "benefit", "application_domain"]].dropna()

        # Construct text = feature + benefit
        df["text"] = df["feature"].astype(str) + " " + df["benefit"].astype(str)

        # Extract labels
        raw_labels = df["application_domain"].tolist()

        # Sort class names alphabetically for reproducibility
        self.class_names = sorted(list(set(raw_labels)))

        # Encode labels
        self.label_encoder = {label: idx for idx, label in enumerate(self.class_names)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}

        self.labels = np.array([self.label_encoder[label] for label in raw_labels])

        self.texts = df["text"].tolist()

        return self

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    def get_class_names(self):
        return self.class_names

    def get_class_distribution(self):
        from collections import Counter
        counts = Counter(self.labels)
        return {
            self.label_decoder[idx]: count for idx, count in counts.items()
        }