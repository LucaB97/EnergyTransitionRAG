import numpy as np
from sentence_transformers import CrossEncoder


class RelevanceProfiler:
    """
    Reranks and profiles retrieved document chunks for a given natural language query
    using a cross-encoder.

    Attributes:
        encoder (CrossEncoder): Cross-encoder model for scoring question-chunk pairs
        floor (float): Minimum score to consider a chunk as relevant (floor safeguard)
    """
    
    def __init__(self, model_name: str, floor: float = 0.25):
        self.encoder = CrossEncoder(model_name)
        self.floor = floor

    
    def score(self, question, chunks):
        """
        Compute cross-encoder relevance scores for each chunk with respect to a query.

        Args:
            question (str): The natural language question to retrieve evidence for.
            chunks (list[dict]): List of chunk dictionaries, each with at least a "text" field.

        Returns:
            np.ndarray: Array of float relevance scores (one per chunk).
        """

        pairs = [(question, c["text"]) for c in chunks]
        scores = self.encoder.predict(pairs)
        return np.array(scores)

    
    def rerank(self, chunks, scores):
        """
        Rerank chunks in descending order of relevance and annotate metadata.

        Args:
            chunks (list[dict]): List of chunk dictionaries (will be copied internally)
            scores (list[float] or np.ndarray): Cross-encoder scores for each chunk

        Returns:
            list[dict]: List of chunk dictionaries, each augmented with:
                - 'final_score': float, cross-encoder score
                - 'final_rank': int, position in descending order of score (1 = highest)
        """
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )
        for rank, (chunk, score) in enumerate(ranked, start=1):
            chunk["final_score"] = float(score)
            chunk["final_rank"] = rank
            # chunk["strong_hit"] = score >= self.floor
        return [c for c, s in ranked]


    def classify_evidence(self, chunks):
        """
        Classify the overall evidence distribution based on chunk scores
        and source diversity.

        Args:
            chunks (list[dict]): Retrieved chunks, each with:
                - "score": cross-encoder score
                - "paper_id": identifier of source paper

        Returns:
            str: Evidence label
        """

        if not chunks:
            return "absent"

        scores = np.array([c["final_score"] for c in chunks])
        paper_ids = [c["paper_id"] for c in chunks]

        mean = scores.mean()
        std = scores.std()
        max_score = scores.max()

        # --- Absolute floor safeguard ---
        if max_score < self.floor:
            return "absent"

        # --- Z-normalization ---
        if std < 1e-6:
            z = np.zeros_like(scores)
        else:
            z = (scores - mean) / std

        strong_indices = np.where(z > 1.0)[0]
        moderate_indices = np.where(z > 0.5)[0]

        strong_hits = len(strong_indices)
        moderate_hits = len(moderate_indices)

        distinct_strong_sources = len(
            set(paper_ids[i] for i in strong_indices)
        )

        max_z = z.max()

        # --- Isolated ---
        if strong_hits == 1 and max_z > 2:
            return "isolated"

        # --- Robust (much stricter now) ---
        if strong_hits >= 10 and distinct_strong_sources >= 3:
            return "robust"

        # --- Thematic ---
        if strong_hits >= 5 and distinct_strong_sources >= 2:
            return "thematic"

        # --- Fragmented ---
        if strong_hits <= 2 and moderate_hits <= 3:
            return "fragmented"

        return "weak"



    def rerank_and_profile(self, question, chunks):
        """
        Convenience method to combine scoring, reranking, and evidence profiling.
        """
        scores = self.score(question, chunks)
        ranked_chunks = self.rerank(chunks, scores)
        evidence_label = self.classify_evidence(ranked_chunks)
        
        return {
            "ranked_chunks": ranked_chunks,
            "evidence_label": evidence_label,
            "metrics": {
                # "scores": scores.tolist(),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "max": float(scores.max()),
                "strong_hits": int(np.sum((scores - scores.mean()) / max(scores.std(),1e-6) > 1))
            }
        }
    

    def get_strong_hits(self, ranked_chunks, z_threshold=1.0):
        """
        Return chunks that are strong hits based on Z-score among the ranked chunks.

        Args:
            ranked_chunks (list[dict]): Chunks already annotated with 'score' (from rerank/profile)
            z_threshold (float): Minimum z-score to consider a chunk a strong hit (default=1.0)

        Returns:
            list[dict]: Chunks with z-score >= z_threshold, annotated with their z-score
        """
        scores = np.array([c["final_score"] for c in ranked_chunks])
        mean = scores.mean()
        std = scores.std()
        if std < 1e-6:
            z_scores = np.zeros_like(scores)
        else:
            z_scores = (scores - mean) / std

        strong_hits = []
        for chunk, z in zip(ranked_chunks, z_scores):
            if z >= z_threshold:
                chunk_copy = chunk.copy()
                chunk_copy["z_score"] = float(z)
                strong_hits.append(chunk_copy)

        return strong_hits