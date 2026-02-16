def determine_grounding_status(metrics):
    used_papers = metrics.get("used_papers", 0)
    paper_dominance = metrics.get("paper_dominance", 1.0)
    multi_source_ratio = metrics.get("multi_source_sentence_ratio", 0.0)

    if used_papers == 0:
        return "incomplete"

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
        "incomplete": 0.6,
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
        f"Evidence structure: {evidence_status}.",
        f"Grounding quality: {grounding_status}."
    ]

    return score, label, explanation




# def compute_confidence(metrics, reason):
#     """
#     Determine a confidence score in [0, 1], a confidence label,
#     and short explanations based on evidence strength.
#     """

#     used_papers = metrics.get("used_papers", 0)
#     paper_dominance = metrics.get("paper_dominance", 1.0)
#     multi_source_ratio = metrics.get("multi_source_sentence_ratio", 0.0)

#     score = 1.0
#     signals = []

#     # --- Evidence sufficiency (absolute, not relative) ---
#     if used_papers == 0:
#         score -= 0.7
#         signals.append(
#             "The answer is not directly supported by retrieved research papers."
#         )
#     elif used_papers < 2:
#         score -= 0.4
#         signals.append(
#             "The answer relies on very limited research evidence."
#         )
#     elif used_papers < 4:
#         score -= 0.2
#         signals.append(
#             "The answer is supported by a small number of research papers."
#         )

#     # --- Evidence robustness ---
#     if paper_dominance > 0.6:
#         score -= 0.3
#         signals.append(
#             "The synthesis relies heavily on a single paper."
#         )
#     elif paper_dominance > 0.4:
#         score -= 0.15
#         signals.append(
#             "One paper contributes more heavily than others."
#         )

#     if multi_source_ratio == 0:
#         score -= 0.4
#         signals.append(
#             "None of the claims are supported by multiple independent sources."
#         )
#     elif multi_source_ratio < 0.3:
#         score -= 0.2
#         signals.append(
#             "Only a small fraction of claims are supported by multiple independent sources."
#         )
#     elif multi_source_ratio > 0.6:
#         score += 0.05  # small bonus for strong corroboration

#     # --- Reason-based cap ---
#     if reason == "insufficient_evidence":
#         score = min(score, 0.4)

#     score = max(0.0, min(1.0, round(score, 2)))

#     # --- Label ---
#     if score >= 0.75:
#         label = "High"
#     elif score >= 0.45:
#         label = "Medium"
#     else:
#         label = "Low"

#     # --- Explanation ---
#     if label == "High":
#         explanation = [
#             "The answer is supported by multiple independent sources with sufficient and balanced evidence."
#         ]
#     else:
#         explanation = signals[:3]

#     return score, label, explanation