import numpy as np
from collections import Counter


class PromiseDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.texts = []
        self.labels = []
        self.class_names = None
        self.label_encoder = None
        self.label_decoder = None

    def load(self):
        data_section = False

        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.lower() == "@data":
                    data_section = True
                    continue

                if not data_section:
                    continue

                # Format: id, 'text', class
                # We split only first and last comma carefully
                first_comma = line.find(",")
                last_comma = line.rfind(",")

                text_part = line[first_comma + 1:last_comma].strip()
                label_part = line[last_comma + 1:].strip()

                # Remove surrounding quotes
                if text_part.startswith("'") and text_part.endswith("'"):
                    text_part = text_part[1:-1]

                self.texts.append(text_part)
                self.labels.append(label_part)

        self.class_names = sorted(list(set(self.labels)))

        # Encode labels
        self.label_encoder = {label: idx for idx, label in enumerate(self.class_names)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}

        self.labels = np.array([self.label_encoder[label] for label in self.labels])

        return self

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    def get_class_names(self):
        return self.class_names

    def get_class_distribution(self):
        counts = Counter(self.labels)
        distribution = {
            self.label_decoder[idx]: count for idx, count in counts.items()
        }
        return distribution
