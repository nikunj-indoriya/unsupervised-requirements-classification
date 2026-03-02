from sentence_transformers import SentenceTransformer
import numpy as np


class SRoBERTaEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("all-distilroberta-v1")

    def encode(self, texts):
        return np.array(self.model.encode(texts, batch_size=32, show_progress_bar=True))
