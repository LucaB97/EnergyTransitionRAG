import faiss
from collections import defaultdict


class SemanticRetriever:
    """
    Semantic retriever for document chunks using vector similarity search.

    This class wraps a FAISS index and an embedding function to retrieve
    relevant text chunks for a given natural language query. 
    It supports retrieval for downstream LLM synthesis.
    """

    def __init__(self, index, chunks, embedding_fn):
        self.index = index
        self.chunks = chunks
        self.embedding_fn = embedding_fn

    
    def search(self, query, top_k=10):
        """
        Retrieve the most semantically similar chunks for a given query.

        This method is recall-oriented: it may return multiple chunks
        from the same document, which is useful when providing rich
        context to an LLM.

        Args:
            query (str): Natural language query.
            top_k (int): Number of chunks to retrieve.

        Returns:
            list[dict]: Retrieved chunk dictionaries with metadata.
        """

        query_embedding = self.embedding_fn(query).astype("float32")
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            chunk["rank"] = rank
            results.append(chunk)

        return results
    

    def display(self, results, max_chars=300):
        """
        Display retrieved chunks grouped by source paper, highlighting how many
        relevant passages were found per paper.

        This method improves interpretability of retrieval results by:
        - Grouping multiple retrieved chunks originating from the same paper
        - Displaying each paper only once in the main index
        - Explicitly indicating how many relevant passages were retrieved per paper
        - Printing the content of each relevant passage separately

        Args:
            results (list[dict]): List of retrieved chunks, where each chunk
                contains at least the following fields:
                - paper_id (str)
                - authors (str)
                - title (str)
                - year (int)
                - text (str)
            max_chars (int, optional): Maximum number of characters to display
                for each passage. Longer passages are truncated for readability.

        Output format:
            [1] Paper Title (Year) — N relevant passages
            └ Passage 1:
                <text snippet>

            └ Passage 2:
                <text snippet>

        Notes:
            This representation aligns with the retrieval logic used for synthesis,
            making it explicit when multiple passages from the same source contribute
            to the final answer. It also avoids misleading index jumps that can occur
            when deduplication is applied without grouping.
        """
        
        papers = defaultdict(list)

        # Group chunks by paper
        for r in results:
            papers[r["paper_id"]].append(r)

        display_idx = 1

        for paper_id, chunks in papers.items():
            authors = chunks[0]["authors"]
            title = chunks[0]["title"]
            year = chunks[0]["year"]
            n_chunks = len(chunks)

            print(f"\n[{display_idx}] {authors} ({year}) — {title}\n  ({n_chunks} relevant passage{'s' if n_chunks > 1 else ''})")

            for i, c in enumerate(chunks, 1):
                print(f"  └ Passage {i}:")
                print("   ", c["text"][:max_chars] + ("..." if len(c["text"]) > max_chars else ""))
                print()

            display_idx += 1