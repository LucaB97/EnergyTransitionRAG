def deduplicate(chunks):
    seen = set()
    unique = []

    for chunk in chunks:
        if chunk.chunk_id not in seen:
            unique.append(chunk)
            seen.add(chunk.chunk_id)

    return unique


def needs_retry(semantic_flags, evidence_flags):
    if semantic_flags["weak_semantic_match"] and not evidence_flags["absent"]:
        return True

    if evidence_flags["isolated"] or evidence_flags["low_density"]:
        return True

    return False 


def assign_limitations(weak_semantic_match=False, absent=False, isolated=False):
    
    if weak_semantic_match:
        return ["The literature does not address this question directly"]
    
    if absent:
        return ["No sufficiently relevant evidence was identified"]
    
    if isolated:
        ["The retrieved evidence is too narrow and context-specific to support synthesis across studies"]