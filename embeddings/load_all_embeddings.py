from embeddings.embedding_manager import EmbeddingManager
from embeddings.transformer_embedder import TransformerEmbedder
from embeddings.word_embedder import WordEmbedder
from preprocessing.text_cleaner import TextCleaner


def get_embedding(model_name, texts, dataset_name):

    manager = EmbeddingManager(dataset_name)
    cached = manager.load(model_name)

    if cached is not None:
        print(f"Loaded cached embedding: {model_name}")
        return cached

    cleaner = TextCleaner()
    texts_clean = [cleaner.clean(t) for t in texts]

    # ==============================
    # TRANSFORMER MODELS
    # ==============================
    transformer_models = {
        "sbert": "paraphrase-MiniLM-L6-v2",
        "sroberta": "all-distilroberta-v1",
        "mpnet": "all-mpnet-base-v2",
        "e5": "intfloat/e5-base-v2",
        "instructor": "hkunlp/instructor-base"
    }

    # ==============================
    # PRETRAINED WORD MODELS
    # ==============================
    word_pretrained_models = {
        "w2v_pretrained": "pretrained/GoogleNews-vectors-negative300.bin",
        "fasttext_pretrained": "pretrained/wiki-news-300d-1M.vec",
        "glove_pretrained": "pretrained/glove.6B.300d.txt"
    }

    # ------------------------------
    # TRANSFORMERS
    # ------------------------------
    if model_name in transformer_models:

        embedder = TransformerEmbedder(
            transformer_models[model_name]
        )
        embeddings = embedder.encode(texts_clean)

    # ------------------------------
    # WORD2VEC SELF
    # ------------------------------
    elif model_name == "w2v_self":

        embedder = WordEmbedder(
            mode="self",
            texts=texts_clean
        )
        embeddings = embedder.encode(texts_clean)

    # ------------------------------
    # PRETRAINED WORD EMBEDDINGS
    # ------------------------------
    elif model_name in word_pretrained_models:

        embedder = WordEmbedder(
            mode="pretrained",
            model_path=word_pretrained_models[model_name]
        )
        embeddings = embedder.encode(texts_clean)

    else:
        raise ValueError(f"Unknown embedding: {model_name}")

    manager.save(model_name, embeddings)
    return embeddings