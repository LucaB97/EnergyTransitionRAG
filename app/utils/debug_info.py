from collections import defaultdict


def get_debug_info(retrieved_chunks, used_chunk_ids):
    """
    Build structured debug information for evidence tracing.
    """

    chunks = []
    paper_stats = defaultdict(lambda: {
        "chunks_retrieved": 0,
        "chunks_used": 0,
        "title": None,
        "authors": None,
        "year": None,
    })

    for rank, c in enumerate(retrieved_chunks, start=1):
        used = c["chunk_id"] in used_chunk_ids

        chunks.append({
            "chunk_id": c["chunk_id"],
            "paper_id": c["paper_id"],
            "title": c.get("title"),
            "authors": c.get("authors"),
            "year": c.get("year"),
            "text": c.get("text"),
            "similarity": c.get("score"),   # if available
            "rank": rank,
            "used_in_synthesis": used,
        })

        p = paper_stats[c["paper_id"]]
        p["chunks_retrieved"] += 1
        p["chunks_used"] += int(used)
        p["title"] = c.get("title")
        p["authors"] = c.get("authors")
        p["year"] = c.get("year")

    retrieved = len(retrieved_chunks)
    used = len(used_chunk_ids)

    unique_papers_retrieved = len(paper_stats)
    unique_papers_used = sum(
        1 for p in paper_stats.values() if p["chunks_used"] > 0
    )

    # paper dominance: fraction of used chunks from most-used paper
    max_chunks_from_one_paper = max(
        (p["chunks_used"] for p in paper_stats.values()),
        default=0
    )

    paper_dominance = (
        max_chunks_from_one_paper / used if used > 0 else 0.0
    )

    return {
        "chunks": chunks,
        "papers": [
            {
                "paper_id": pid,
                **stats
            }
            for pid, stats in paper_stats.items()
        ],
        "metrics": {
            "retrieved_chunks": retrieved,
            "used_chunks": used,
            "chunk_coverage": used / retrieved if retrieved else 0.0,

            "retrieved_papers": unique_papers_retrieved,
            "used_papers": unique_papers_used,
            "paper_coverage": (
                unique_papers_used / unique_papers_retrieved
                if unique_papers_retrieved else 0.0
            ),

            "paper_dominance": round(paper_dominance, 3),
        }
    }
