def compute_confidence(metrics, reason):
    """
    Determine a confidence score in [0, 1] and a relative confidence label, 
    based on evidence strength and failure modes.
    """

    # Hard failure modes
    if reason == "out_of_scope":
        return 0.0, "Low"


    paper_coverage = metrics.get("paper_coverage", 0.0)
    paper_dominance = metrics.get("paper_dominance", 1.0)
    multi_source_ratio = metrics.get("multi_source_sentence_ratio", 0.0)


    score = 1.0
    
    score -= 0.5 * (1 - paper_coverage) # low-breadth penalty
    score -= 0.3 * paper_dominance # over-reliance penalty
    score -= 0.2 * (1 - multi_source_ratio) # lack-of-corroboration penalty
    
    score = max(0.0, round(score, 2))


    # Adjust score to reflect failure modes
    if reason == "insufficient_evidence":
        score = min(score, 0.4)


    #Label
    if score >= 0.75:
        label = "High"
    elif score >= 0.45:
        label = "Medium"
    else:
        label = "Low"

    return score, label