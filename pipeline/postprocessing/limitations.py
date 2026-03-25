def assign_limitations(semantic_alignment_score, absent=False, low_density=False):
    
    if semantic_alignment_score < 0.25:
        return ["The literature does not address this question directly"]
    
    if absent:
        return ["No sufficiently relevant evidence was identified"]
    
    if low_density:
        return ["The retrieved evidence is too narrow and context-specific to support synthesis across studies"]