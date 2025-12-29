import numpy as np
import faiss


def build_faiss_index(chunks, embedding_fn, index_path=None, embeddings_path=None):
    """
    Build a FAISS index for fast similarity search using cosine similarity.
    A list of text chunks are embedded using a user-provided function.
    The embeddings are L2-normalized so that inner product corresponds
    to cosine similarity. 
    A FAISS index is built over the embeddings for efficient similarity search.
    Optionally persist the index to disk for reuse.

    Args:
        chunks (list[dict]): List of text chunks to be indexed. Each chunk
            must contain a 'text' field used for embedding. Additional metadata
            is preserved externally and not stored in the FAISS index.
        embedding_fn (Callable[[str], np.ndarray]): Function that takes a text
            string and returns a 1D NumPy embedding vector.
        index_path (str or Path, optional): If provided, the FAISS index
            is saved to this path after construction.

    Returns:
        tuple:
            - faiss.Index: FAISS index containing all embeddings
            - np.ndarray: 2D array of normalized embeddings (n_chunks, dim)
    """

    texts = [c["text"] for c in chunks]
    embeddings = embedding_fn(texts)
    embeddings = embeddings.astype("float32")
    
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    if index_path is not None:
        faiss.write_index(index, str(index_path))
    
    if embeddings_path is not None:
        np.save(embeddings_path, embeddings)
    
    return index, embeddings


def load_faiss(path):
    """
    Load a FAISS index from disk.

    Args:
        path (str or Path): Path to the saved FAISS index.

    Returns:
        faiss.Index: Loaded FAISS index.
    """

    return faiss.read_index(str(path))