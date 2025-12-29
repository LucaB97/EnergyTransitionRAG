import numpy as np


_model_cache = {}

def hf_embedding(texts, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings for one or more texts using a SentenceTransformer model.

    Args:
        texts (str | list[str]): Input text(s) to embed.
        model_name (str): Name of the SentenceTransformer model.

    Returns:
        np.ndarray: 2D NumPy array of shape (n_texts, embedding_dim)
    """

    from sentence_transformers import SentenceTransformer

    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)

    if isinstance(texts, str):
        texts = [texts]

    return _model_cache[model_name].encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )
