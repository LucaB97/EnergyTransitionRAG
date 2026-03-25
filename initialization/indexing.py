import json
from services.embeddings import HFEmbedding, OpenAIEmbedding
from services.indexing import build_faiss_index


def build_index_pipeline(config, chunks_path, index_path):
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if config.embedding == "hf":
        embedding_fn = HFEmbedding()
    else:
        embedding_fn = OpenAIEmbedding()

    index, embeddings = build_faiss_index(chunks, embedding_fn, index_path=index_path)

    return index, chunks