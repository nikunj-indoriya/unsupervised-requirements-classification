from sentence_transformers import SentenceTransformer
import numpy as np


class TransformerEmbedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return np.array(
            self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True
            )
        )