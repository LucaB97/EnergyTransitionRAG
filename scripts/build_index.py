import json
from utils.embeddings import hf_embedding, openai_embedding
from utils.indexing import build_faiss_index

with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# build_faiss_index(chunks, hf_embedding, index_path="data/faiss.index", embeddings_path="data/embeddings.npy")
build_faiss_index(chunks, openai_embedding, index_path="data/openai_faiss.index", embeddings_path="data/openai_embeddings.npy")

# print(f"Indexed {len(texts)} chunks")

