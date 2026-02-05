import numpy as np
import os

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



def openai_embedding(texts, model="text-embedding-3-small", batch_size=200):
    """
    Generate embeddings for one or more texts using OpenAI embeddings.

    Args:
        texts (str | list[str]): Input text(s) to embed.
        model (str): OpenAI embedding model.
        batch_size (int): Number of texts per API call.

    Returns:
        np.ndarray: 2D NumPy array of shape (n_texts, embedding_dim)
    """
    
    from openai import OpenAI
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if isinstance(texts, str):
        texts = [texts]

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        all_embeddings.extend([item.embedding for item in response.data])

    return np.array(all_embeddings, dtype="float32")
