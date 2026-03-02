import tensorflow_hub as hub
import numpy as np


class USEEmbedder:
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def encode(self, texts):
        embeddings = self.model(texts)
        return np.array(embeddings)
