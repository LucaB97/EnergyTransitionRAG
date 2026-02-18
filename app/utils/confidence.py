def determine_grounding(metrics):

    used_papers = metrics.get("used_papers", 0)
    dominance = metrics.get("paper_dominance", 1.0)
    multi_ratio = metrics.get("multi_source_sentence_ratio", 0.0)

    if used_papers == 0:
        return 0.0, "not_answered"
    
    if used_papers >= 3:
        base = 0.6
    elif used_papers == 2:
        base = 0.5
    else:
        base = 0.35

    dominance_penalty = max(0, dominance - 0.5) * 0.5
    corroboration_bonus = multi_ratio * 0.35

    score = base + corroboration_bonus - dominance_penalty
    score = max(0.05, min(0.9, score))
    
    if score >= 0.85:
        label = "complete" #explicit corroboration + balance
    elif score >= 0.65:
        label = "strong" #balanced multi-source
    elif score >= 0.45:
        label = "partial" #multi-source but low corroboration
    elif score >= 0.25:
        label = "weak" #mono or imbalanced multi
    else:
        label = "very_weak" #mono-source, dominant

    return score, label



def compute_confidence(pipeline_status, evidence_structure, grounding_quality, grounding_score=None):

    if pipeline_status != "success" or grounding_quality == "not_applicable" or grounding_score is None:
        return 0.0, "None", ["Not applicable"]

    structural_base = {
        "absent": 0.0,
        "isolated": 0.25,
        "weak": 0.45,
        "fragmented": 0.6,
        "thematic": 0.75,
        "robust": 0.85
    }[evidence_structure]

    score = min(structural_base, grounding_score)
    score = max(0.0, min(0.9, score))

    # Label mapping
    if score > 0.80:
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
        f"evidence structure: {evidence_structure}",
        f"grounding quality: {grounding_quality}"
    ]

    return score, label, explanation