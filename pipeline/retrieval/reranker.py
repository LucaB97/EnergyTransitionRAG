class RelevanceProfiler:
    """
    Wrapper that uses a pluggable reranker.
    """

    def __init__(self, reranker):
        if not hasattr(reranker, "rerank"):
            raise ValueError("reranker must implement a .rerank(query, chunks) method")  
        self.reranker = reranker

    def rerank(self, query, chunks):
        return self.reranker.rerank(query, chunks)
