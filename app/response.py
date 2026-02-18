from app.schemas import QueryResponse

def build_response(
    question,
    pipeline_status,
    evidence,
    grounding,
    limitations,
    meta=None,
    confidence=None,
    debug=None
):
    return QueryResponse(
        question=question,
        pipeline_status=pipeline_status,
        evidence_structure=evidence,
        grounding_quality=grounding,
        answer=[],
        limitations=limitations,
        sources=[],
        meta=meta or {},
        confidence=confidence,
        debug=debug or {}
    )
