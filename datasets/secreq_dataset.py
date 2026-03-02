import pandas as pd
import numpy as np
from collections import Counter


class SecReqDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.texts = []
        self.labels = []
        self.class_names = None
        self.label_encoder = None
        self.label_decoder = None

    def load(self):

        df = pd.read_csv(self.file_path)

        self.texts = df["text"].astype(str).tolist()
        raw_labels = df["label"].astype(str).tolist()

        # Sort alphabetically for reproducibility
        self.class_names = sorted(list(set(raw_labels)))

        self.label_encoder = {
            label: idx for idx, label in enumerate(self.class_names)
        }

        self.label_decoder = {
            idx: label for label, idx in self.label_encoder.items()
        }

        self.labels = np.array(
            [self.label_encoder[label] for label in raw_labels]
        )

        return self

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    def get_class_names(self):
        return self.class_names

    def get_class_distribution(self):
        counts = Counter(self.labels)
        return {
            self.label_decoder[idx]: count
            for idx, count in counts.items()
        }