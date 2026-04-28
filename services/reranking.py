import numpy as np

class FlashRankReranker(BaseReranker):
    def __init__(self, model_name="ms-marco-TinyBERT-L-2-v2", floor=0.0):
        from flashrank import Ranker, RerankRequest
      
        self.ranker = Ranker(model_name=model_name)
        self.floor = floor

    def rerank(self, query, chunks):
        if not chunks:
            return []

        passages = [{"text": c["text"], "meta": i} for i, c in enumerate(chunks)]

        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        ranked = []
        for rank, r in enumerate(results, start=1):
            c = chunks[r["meta"]]
            c["final_score"] = float(max(r["score"], self.floor))
            c["final_rank"] = rank
            ranked.append(c)

        return ranked


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", floor=0.25):
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = CrossEncoder(model_name, device=device)
        self.floor = floor

    def rerank(self, query, chunks):
        if not chunks:
            return []

        pairs = [(query, c["text"]) for c in chunks]
        scores = self.encoder.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        output = []
        for rank, (c, score) in enumerate(ranked, start=1):
            c["final_score"] = float(max(score, self.floor))
            c["final_rank"] = rank
            output.append(c)

        return output
