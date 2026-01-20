def compute_confidence(metrics):
    """
    Compute a confidence score in [0, 1] based on evidence strength.
    """

    score = 0.0
    weight_sum = 0.0

    def add(value, weight):
        nonlocal score, weight_sum
        score += weight * max(0.0, min(1.0, value))
        weight_sum += weight

    add(metrics.get("paper_coverage", 0.0), 0.35)
    add(1 - metrics.get("paper_dominance", 0.0), 0.25)
    add(metrics.get("multi_source_sentence_ratio", 0.0), 0.25)
    add(metrics.get("chunk_coverage", 0.0), 0.15)

    return score / weight_sum if weight_sum else 0.0