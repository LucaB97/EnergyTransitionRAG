def deduplicate(chunks):
    seen = set()
    unique = []

    for chunk in chunks:
        if chunk.chunk_id not in seen:
            unique.append(chunk)
            seen.add(chunk.chunk_id)

    return unique


def needs_retry(evidence_flags):
    if evidence_flags["weak_semantic_match"] and not evidence_flags["absent"]:
        return True

    if evidence_flags["isolated"] or evidence_flags["low_density"]:
        return True

    return False 


def assign_limitations(label):
    
    if label=="weak_semantic_match":
        return ["The literature does not address this question directly"]
    
    if label=="absent":
        return ["No sufficiently relevant evidence was identified"]
    
    if label=="isolated":
        ["The retrieved evidence is too narrow and context-specific to support synthesis across studies"]