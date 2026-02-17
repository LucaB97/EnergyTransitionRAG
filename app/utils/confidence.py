def determine_grounding_status(metrics):
    used_papers = metrics.get("used_papers", 0)
    paper_dominance = metrics.get("paper_dominance", 1.0)
    multi_source_ratio = metrics.get("multi_source_sentence_ratio", 0.0)

    if used_papers == 0:
        return "not_answered"

    if used_papers < 2:
        return "weak"

    if paper_dominance > 0.7:
        return "weak"

    if multi_source_ratio == 0:
        return "weak"

    return "complete"



def compute_confidence(pipeline_status, evidence_status, grounding_status, metrics=None):

    if pipeline_status != "success":
        return 0.0, "None", ["Technical failure during processing."]

    structural_base = {
        "absent": 0.0,
        "isolated": 0.2,
        "weak": 0.4,
        "fragmented": 0.6,
        "thematic": 0.8,
        "robust": 0.95
    }[evidence_status]

    grounding_factor = {
        "complete": 1.0,
        "weak": 0.8,
        "not_answered": 0.6,
        "not_applicable": 1.0
    }[grounding_status]

    # Optional small refinement from metrics
    multi_source_ratio = metrics.get("multi_source_sentence_ratio", 0.0)

    corroboration_bonus = 0.05 if multi_source_ratio > 0.6 else 0.0

    score = structural_base * grounding_factor
    score += corroboration_bonus
    score = max(0.0, min(1.0, round(score, 2)))

    # Label mapping
    if score >= 0.85:
        label = "Very High"
    elif score >= 0.7:
        label = "High"
    elif score >= 0.5:
        label = "Moderate"
    elif score >= 0.3:
        label = "Low"
    elif score > 0:
        label = "Very Low"
    else:
        label = "None"

    explanation = [
        f"evidence structure: {evidence_status}",
        f"grounding quality: {grounding_status}"
    ]

    return score, label, explanation