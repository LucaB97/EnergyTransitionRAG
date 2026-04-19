import json
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import random
import time

from services.embeddings import OpenAIEmbedding
from services.indexing import load_faiss
from services.llm_clients import OpenAIClient

from pipeline.preprocessing.normalization import Normalizer

from initialization.config import DEFAULT_CONFIG
from pipeline.retrieval.retriever import SemanticRetriever, BM25Retriever, HybridRetriever
from pipeline.retrieval.reranker import RelevanceProfiler
from pipeline.llm.relevance_evaluation import EvidenceRelevanceJudge

from utils.prompt import SINGLE_RELEVANCE_JUDGE_PROMPT, GROUP_RELEVANCE_JUDGE_PROMPT, SINGLE_RELEVANCE_JUDGE_PROMPT_BIN, GROUP_RELEVANCE_JUDGE_PROMPT_BIN

load_dotenv()
config = DEFAULT_CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_dir = PROJECT_ROOT / "data"
chunks_path = data_dir / f"chunks_{config.chunk_size}t_{config.overlap}o.json"
index_path = data_dir / f"faiss_{config.embedding}_{config.chunk_size}t_{config.overlap}o.index"
topN = 15

with open(chunks_path, encoding="utf-8") as f:
    chunks = json.load(f)

index = load_faiss(index_path)
embedding_fn = OpenAIEmbedding()

normalizer = Normalizer(use_lemmatization=False)
llm = OpenAIClient(model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.0)


semantic_retriever = SemanticRetriever(index, chunks, embedding_fn)
bm25_retriever = BM25Retriever(chunks, normalizer)
retriever = HybridRetriever(semantic_retriever, bm25_retriever)
relevance_profiler = RelevanceProfiler()
judge = EvidenceRelevanceJudge(llm)
topk_faiss, topk_bm25 = 30, 30

user_queries = ["Impact of renewable energy adoption on air quality",
                "How do ownership models shape social outcomes of renewable projects?",
                "Influence of social networks on adoption of solar panels",
                "Impact of political orientations on renewable energy perception",
                "Economic effects of community-owned renewable energy initiatives"
                ]


# individual_runs_aggr_time = 0
unified_runs_aggr_time = 0
counter = 0

for query in user_queries:
    print(f"\n{query}")
    norm_query = normalizer.normalize(query)
    retrieved_chunks = retriever.search(query, norm_query, topk_faiss=topk_faiss, topk_bm25=topk_bm25)
    reranked_chunks = relevance_profiler.rerank(query, retrieved_chunks)
    top_chunks = reranked_chunks[:topN]
    texts = [chunk["text"] for chunk in top_chunks]

    for j in range(5):
        counter += 1

        start_time = time.perf_counter()
        scores = judge.judge_relevance_unified(query, texts, GROUP_RELEVANCE_JUDGE_PROMPT_BIN)
        unified_runs_aggr_time += time.perf_counter() - start_time
        avg_score = sum(scores) / (len(scores))

        print(f"{avg_score}")

print(f"Avg runtime\nUnified llm calls: {unified_runs_aggr_time / counter}")
# print(f"Avg runtime\nIndividual llm calls: {individual_runs_aggr_time / counter}\nUnified llm calls: {unified_runs_aggr_time / counter}")



# individual_runs_aggr_time = 0
# unified_runs_aggr_time = 0

# for query in user_queries:
#     print(query)
#     norm_query = normalizer.normalize(query)
#     retrieved_chunks = retriever.search(query, norm_query, topk_faiss=topk_faiss, topk_bm25=topk_bm25)
#     reranked_chunks = relevance_profiler.rerank(query, retrieved_chunks)
#     top_chunks = reranked_chunks[:topN]
    
#     for j in range(3):
#         print(f"RUN {j+1}")

#         indexed_passages = list(enumerate(top_chunks))
#         random.shuffle(indexed_passages)
#         shuffled_passages = [p for _, p in indexed_passages]
#         texts = [p["text"] for p in shuffled_passages]

#         print("Individual runs; Binary scores")
#         start_time = time.perf_counter()
#         scores_list = []
#         for text in texts:
#             score = judge.judge_relevance(query, text, SINGLE_RELEVANCE_JUDGE_PROMPT_BIN)
#             scores_list.append(int(score))
#         individual_runs_aggr_time += time.perf_counter() - start_time
#         avg_score = sum(scores_list) / (len(scores_list))
#         print(f"Scores: {scores_list} -> {avg_score}")


#         print("Individual runs; Ternary scores")
#         start_time = time.perf_counter()
#         scores_list = []
#         for text in texts:
#             score = judge.judge_relevance(query, text, SINGLE_RELEVANCE_JUDGE_PROMPT)
#             scores_list.append(int(score))
#         individual_runs_aggr_time += time.perf_counter() - start_time
#         avg_score = sum(scores_list) / (2 * len(scores_list))
#         print(f"Scores: {scores_list} -> {avg_score}")


        # print("Unified run; Binary scores")
        # start_time = time.perf_counter()
        # scores = judge.judge_relevance_unified(query, texts, GROUP_RELEVANCE_JUDGE_PROMPT_BIN)
        # unified_runs_aggr_time += time.perf_counter() - start_time
        # avg_score = sum(scores) / (len(scores))
        # print(f"Scores: {scores} -> {avg_score}")


#         print("Unified run; Ternary scores")
#         start_time = time.perf_counter()
#         scores = judge.judge_relevance_unified(query, texts, GROUP_RELEVANCE_JUDGE_PROMPT)
#         unified_runs_aggr_time += time.perf_counter() - start_time
#         avg_score = sum(scores) / (2 * len(scores))
#         print(f"Scores: {scores} -> {avg_score}")


# num_tests = len(user_queries) * (j+1)
# print(f"Avg runtime\nIndividual llm calls: {individual_runs_aggr_time / (num_tests)}\nUnified llm calls: {unified_runs_aggr_time / (num_tests)}")