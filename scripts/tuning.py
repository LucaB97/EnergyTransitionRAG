from pathlib import Path
import argparse
import json
import numpy as np
from utils.embeddings import OpenAIEmbedding
from utils.indexing import load_faiss
from utils.retriever import SemanticRetriever, BM25Retriever, HybridRetriever
from utils.cross_encoder import RelevanceProfiler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUERIES = [
    "Social acceptance of community wind projects in areas with low electricity demand",
    "Effects of minor policy changes on household energy literacy programs",
    "Participation of informal local groups in renewable energy initiatives",
    "Influence of neighborhood social networks on adoption of solar panels in small towns",
    "Energy poverty considerations in communities with high off-grid adoption",
    "Challenges of promoting energy literacy in transient communities",
    "Influence of minor local regulations on small-scale renewable energy projects",
    "Participation of volunteers in energy transition awareness campaigns in sparsely populated regions",

    "Household affordability and adoption of solar energy in low-income communities",
    "Community engagement strategies in renewable energy projects",
    "Public attitudes toward wind farms in semi-urban areas",
    "Energy literacy programs for promoting energy efficiency",
    "Distributional effects of subsidies for residential solar panels",
    "Trust in local authorities influencing renewable energy adoption",
    "Equity in decision-making for local renewable energy projects",
    "Economic well-being effects of community-owned renewable energy initiatives",
    "Public perception of renewable energy policies in minority communities",
    "Participation of local stakeholders in planning renewable energy infrastructure",
    "Community engagement in microgrid implementation projects",
    "Impact of gender dynamics on participation in renewable energy cooperatives",
    "Effect of municipal communication strategies on local renewable energy awareness",
    
    "Effectiveness of participatory decision-making on social acceptance of onshore wind projects",
    "Impact of targeted energy literacy programs on adoption of household solar PV in disadvantaged neighborhoods",
    "Relationship between energy poverty alleviation and distributed renewable energy schemes in rural communities",
    "Role of trust in institutions on equitable energy transition outcomes",
    "Influence of social norms on community investment in renewable energy cooperatives",
    "Governance frameworks supporting minority inclusion in renewable energy projects",
    "Effectiveness of participatory governance in achieving equitable renewable energy outcomes",
    "Role of social norms and trust in accelerating community-owned solar adoption",
    "Relationship between energy poverty alleviation and inclusive policy design in rural renewable energy programs"
    ]


def semantic_norm(score, a, b):
    import math
    return 1 / (1 + math.exp(-a * (score - b)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_contribution(abs_relevance, z, z_thr=0, kz=1.5):
    rel_relevance = sigmoid(kz * (z - z_thr))
    return abs_relevance * rel_relevance


def normalization_params(queries, retriever, relevance_profiler, parameters):

    all_scores = []

    for i, q in enumerate(queries):
        retrieved_chunks = retriever.search(q, topk_faiss=30, topk_bm25=30)
        reranked_chunks = relevance_profiler.rerank(q, retrieved_chunks)
        scores = [chunk["final_score"] for chunk in reranked_chunks[:15]]
        all_scores.extend(scores)

    all_scores = np.array(all_scores)
    b = np.median(all_scores)
    x90 = np.percentile(all_scores, 90)
    y_target = 0.9
    a = -np.log(1 / y_target - 1) / (x90 - b)

    parameters["normalization_params"] = {
        "a": a,
        "b": b,
        "std_global": all_scores.std()
    }

    return parameters


def contribution_quantiles(queries, retriever, relevance_profiler, parameters, alpha):

    a,b, std_global = (parameters["normalization_params"]["a"], 
                       parameters["normalization_params"]["b"], 
                       parameters["normalization_params"]["std_global"])
    
    sum_contributions_list = []
    all_contributions = []

    for i, q in enumerate(queries):        
        retrieved_chunks = retriever.search(q, topk_faiss=30, topk_bm25=30)
        reranked_chunks = relevance_profiler.rerank(q, retrieved_chunks)
        
        relevant_chunks = reranked_chunks[:15]
        scores = np.array([chunk["final_score"] for chunk in relevant_chunks])    
        mean_score, std_score = scores.mean(), scores.std()

        if std_score < 1e-6:
            z = np.zeros_like(scores)
        else:
            std = max(std_score, std_global*alpha) 
            z = (scores - mean_score) / std
        
        abs_relevance = np.array([semantic_norm(s, a, b) for s in scores])
        contributions = compute_contribution(abs_relevance, z)
        
        all_contributions.extend(contributions)
        sum_contributions_list.append(contributions.sum())

        parameters["contributions_per_query"] = {
            "q10": np.percentile(sum_contributions_list, 10),
            "q25": np.percentile(sum_contributions_list, 25),
            "q50": np.percentile(sum_contributions_list, 50),
            "q75": np.percentile(sum_contributions_list, 75),
            "q90": np.percentile(sum_contributions_list, 90),
            "observed_values": sum_contributions_list,
        }
        
        parameters["chunk_contributions"] = {
            "q10": np.percentile(all_contributions, 10),
            "q25": np.percentile(all_contributions, 25),
            "q50": np.percentile(all_contributions, 50),
        }

    return parameters


def effective_sources_quantiles(queries, retriever, relevance_profiler, parameters, alpha):

    a,b, std_global, min_contribution = (parameters["normalization_params"]["a"], 
                                         parameters["normalization_params"]["b"], 
                                         parameters["normalization_params"]["std_global"],
                                         parameters["chunk_contributions"]["q25"])
    
    effective_sources_list = []

    for i, q in enumerate(queries):       
        retrieved_chunks = retriever.search(q, topk_faiss=30, topk_bm25=30)
        reranked_chunks = relevance_profiler.rerank(q, retrieved_chunks)
        
        relevant_chunks = reranked_chunks[:15]
        paper_ids = [c["paper_id"] for c in relevant_chunks]
        scores = np.array([chunk["final_score"] for chunk in relevant_chunks])    
        mean_score, std_score, max_score = scores.mean(), scores.std(), scores.max()

        if std_score < 1e-6:
            z = np.zeros_like(scores)
        else:
            std = max(std_score, std_global*alpha) 
            z = (scores - mean_score) / std
        
        abs_relevance = np.array([semantic_norm(s, a, b) for s in scores])
        contributions = compute_contribution(abs_relevance, z)

        source_weights = {}
        for i, contrib in enumerate(contributions):
            if contrib > min_contribution:
                pid = paper_ids[i]
                source_weights[pid] = source_weights.get(pid, 0) + contrib

        effective_sources_list.append(len(source_weights))

    parameters["effective_sources_per_query"] = {
        "q10": np.percentile(effective_sources_list, 10),
        "q25": np.percentile(effective_sources_list, 25),
        "q50": np.percentile(effective_sources_list, 50),
        "q75": np.percentile(effective_sources_list, 75),
        "q90": np.percentile(effective_sources_list, 90),
        "observed_values": effective_sources_list,
    }

    return parameters


def tuning(queries, chunks_path, index_path, params_path, alpha_std):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)   

    index = load_faiss(index_path)
    embedding_fn = OpenAIEmbedding()
    semantic_retriever = SemanticRetriever(index, chunks, embedding_fn)
    bm25_retriever = BM25Retriever(chunks)
    retriever = HybridRetriever(semantic_retriever, bm25_retriever)
    relevance_profiler = RelevanceProfiler()
    parameters = {}

    print("Starting tuning...")
    parameters = normalization_params(queries, retriever, relevance_profiler, parameters)
    print("1/3")
    parameters = contribution_quantiles(queries, retriever, relevance_profiler, parameters, alpha_std)
    print("2/3")
    parameters = effective_sources_quantiles(queries, retriever, relevance_profiler, parameters, alpha_std)
    print("3/3")

    # Choose a location in your project
    params_path.parent.mkdir(exist_ok=True)

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=2)

    print(f"Saved parameters to {params_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--embedding", type=str)
    parser.add_argument("--alpha_std", type=str)
    args = parser.parse_args()

    chunk_size = args.chunk_size
    overlap = args.overlap
    embedding = args.embedding
    alpha_std = args.alpha_std

    if chunk_size is None or overlap is None:
        chunk_size = 500
        overlap = 100

    if embedding is None:
        embedding = "openai"

    if alpha_std is None:
        alpha_std = 0.5

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")
    
    if embedding != "hf" and embedding != "openai":
        raise ValueError("Embedding strategy must be \"hf\" or \"openai\"")
    
    chunks_path = PROJECT_ROOT / "data" / f"chunks_{chunk_size}t_{overlap}o.json"
    index_path = PROJECT_ROOT / "data" / f"faiss_{embedding}_{chunk_size}t_{overlap}o.index"
    parameters_path = PROJECT_ROOT / "data" / f"parameters_{embedding}_{chunk_size}t_{overlap}o.json"

    tuning(QUERIES, chunks_path, index_path, parameters_path, alpha_std)


if __name__ == "__main__":
    main()