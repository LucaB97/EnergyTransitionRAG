import argparse
import json
from utils.embeddings import hf_embedding, openai_embedding
from utils.indexing import build_faiss_index


parser = argparse.ArgumentParser(description="Build FAISS index for text chunks")
parser.add_argument("--chunk-size", type=int)
parser.add_argument("--overlap", type=int)
parser.add_argument("--embedding", type=str)
args = parser.parse_args()

chunk_size = args.chunk_size
overlap = args.overlap
embedding = args.embedding

if chunk_size is None or overlap is None:
    chunk_size = 500
    overlap = 100
    
chunks_json = f"data/chunks_{chunk_size}t_{overlap}o.json"

if embedding == "hf":
    strategy = hf_embedding
    path = f"data/faiss_hf_{chunk_size}t_{overlap}o.index"
else:
    strategy = openai_embedding
    path = f"data/faiss_openai_{chunk_size}t_{overlap}o.index"

with open(chunks_json, "r", encoding="utf-8") as f:
    chunks = json.load(f)

build_faiss_index(chunks, strategy, index_path=path)
