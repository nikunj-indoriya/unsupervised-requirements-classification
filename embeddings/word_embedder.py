from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import os


class WordEmbedder:
    def __init__(self, mode, model_path=None, texts=None, vector_size=300):

        self.mode = mode

        if mode == "pretrained":

            # Handle GloVe conversion automatically
            if model_path.endswith(".txt"):
                converted_path = model_path + ".word2vec"

                if not os.path.exists(converted_path):
                    print("Converting GloVe to word2vec format...")
                    glove2word2vec(model_path, converted_path)

                model_path = converted_path
                binary = False
            elif model_path.endswith(".vec"):
                binary = False
            else:
                binary = True

            self.model = KeyedVectors.load_word2vec_format(
                model_path,
                binary=binary
            )

            self.vector_size = self.model.vector_size

        elif mode == "self":

            tokenized = [t.split() for t in texts]

            w2v = Word2Vec(
                sentences=tokenized,
                vector_size=vector_size,
                window=5,
                min_count=1,
                workers=4
            )

            self.model = w2v.wv
            self.vector_size = vector_size

        else:
            raise ValueError("Unknown word embedding mode")

    def encode(self, texts):

        embeddings = []

        for text in texts:
            tokens = text.split()
            vectors = [
                self.model[token]
                for token in tokens
                if token in self.model
            ]

            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))

        return np.array(embeddings)