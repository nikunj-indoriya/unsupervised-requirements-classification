import os
import numpy as np


class EmbeddingManager:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.cache_dir = f"cache/embeddings/{dataset_name}"
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, model_name):
        return os.path.join(self.cache_dir, f"{model_name}.npy")

    def save(self, model_name, embeddings):
        np.save(self.get_cache_path(model_name), embeddings)

    def load(self, model_name):
        path = self.get_cache_path(model_name)
        if os.path.exists(path):
            return np.load(path)
        return None
