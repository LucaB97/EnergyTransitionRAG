import numpy as np


def evaluate_grounding_quality(metrics):
    """
    Evaluate the grounding quality of a generated answer.

    The grounding score reflects how well the answer integrates
    and distributes citations across multiple sources.

    Parameters
    ----------
    metrics : dict
        Dictionary containing grounding-related metrics:
            - used_papers (int): Number of distinct cited papers.
            - paper_dominance (float): Proportion of citations
              attributed to the most frequently cited paper.
            - multi_source_sentence_ratio (float): Fraction of
              sentences supported by multiple sources.

    Returns
    -------
    grounding_score : float
        Continuous grounding quality score in [0, 1].

    flags : dict
        Diagnostic boolean flags describing grounding properties.
    """

    used_papers = metrics.get("used_papers", 0)
    dominance = metrics.get("paper_dominance", 1.0)
    multi_ratio = metrics.get("multi_source_sentence_ratio", 0.0)

    flags = {
        "no_citations": used_papers == 0,
        "single_source_reliance": used_papers == 1,
        "multi_source_grounding": used_papers > 3,
        "high_source_dominance": dominance > 0.7,
        "moderate_source_dominance": 0.4 < dominance <= 0.7,
        "balanced_source_usage": dominance <= 0.4,
        "cross_source_corroboration": multi_ratio >= 0.3,
        "no_corroboration": multi_ratio == 0,
        "low_corroboration": multi_ratio <= 0.2
    }

    if used_papers == 0:
        return 0.0, flags

    # Base score from source count
    if used_papers > 3:
        base = 0.75
    elif used_papers == 3:
        base = 0.60
    elif used_papers == 2:
        base = 0.45
    else:
        base = 0.35

    dominance_penalty = max(0, dominance - 0.5) * 0.5
    corroboration_bonus = multi_ratio * 0.4

    score = base + corroboration_bonus - dominance_penalty
    score = max(0.0, min(1.0, score))

    return score, flags



def explain_grounding(metrics, flags):
    """
    Generate concise, severity-aware explanations for grounding quality.
    """

    # --- Critical ---
    if flags.get("no_citations"):
        return "The answer does not cite any sources"
    
    
    source_usage_bullet = f"Source usage — {metrics['used_papers']} papers\n"
    
    if flags.get("multi_source_grounding"):
        source_usage_bullet += "The synthesis uses evidence from multiple independent sources"
    elif flags.get("single_source_reliance"):
        source_usage_bullet += "The synthesis relies on a single source"
    else:
        source_usage_bullet += "The synthesis relies on a limited number of sources"


    paper_dominance_bullet = f"Source dominance — {metrics['paper_dominance']:.2f}\n"

    if flags.get("single_source_reliance"):
        paper_dominance_bullet += "All of the used evidence comes from a single source"
    elif flags.get("high_source_dominance"):
        paper_dominance_bullet += "Most of the used evidence comes from a single source"
    elif flags.get("moderate_source_dominance"):
        paper_dominance_bullet += "A relevant portion of the used evidence comes from a single source"
    else:
        paper_dominance_bullet += "Citations are fairly distributed across different sources"


    corroboration_bullet = f"Cross-source support — {metrics['multi_source_sentence_ratio']:.2f}\n"
    if flags.get("no_corroboration"):
        corroboration_bullet += "Claims are not corroborated across multiple sources"
    elif flags.get("low_corroboration"):
        corroboration_bullet += "Only limited cross-source corroboration is present"
    else:
        corroboration_bullet += "Several statements are supported by multiple sources"

    return [source_usage_bullet, paper_dominance_bullet, corroboration_bullet]



def evaluate_confidence_profile(pipeline_status, 
                                evidence_score=None,
                                grounding_score=None, grounding_metrics=None, grounding_flags=None,
                                reason=None):
    """
    Compute a multi-axis confidence profile: semantic alignment, evidence structure, grounding quality.
    Each axis gets a score (0-1), a level (Weak/Moderate/Strong), and optional explanations.
    """

    if pipeline_status != "success" or evidence_score is None or grounding_score is None:
        return {
            "status": "Not applicable",
            "reason": reason
        }

    # --- Evidence relevance ---
    if evidence_score >= 0.6:
        evidence_level = "Good"
    elif evidence_score >= 0.3:
        evidence_level = "Limited"
    else:
        evidence_level = "Low"

    evidence_relevance = {
        "level": evidence_level,
        "score": evidence_score,
    }

    # --- Grounding quality ---
    if grounding_score >= 0.75:
        grounding_level = "Strong"
    elif grounding_score >= 0.5:
        grounding_level = "Moderate"
    else:
        grounding_level = "Weak"

    grounding_quality = {
        "level": grounding_level,
        "score": grounding_score,
        "explanation": explain_grounding(grounding_metrics, grounding_flags) if grounding_flags else []
    }

    # --- Assemble ---
    profile = {
        "evidence": evidence_relevance,
        "grounding": grounding_quality,
        "status": "Success",
    }

    return profile