from app.schemas import QueryResponse

def build_response(
    question,
    pipeline_status,
    limitations,
    answer=[],
    sources=[],
    meta=None,
    evidence_structure=None,
    grounding_metrics=None,
    confidence=None,
    trace=None
):
    return QueryResponse(
        question=question,
        pipeline_status=pipeline_status,
        answer=answer,
        limitations=limitations,
        sources=sources,
        meta=meta or {},
        evidence_structure=evidence_structure,
        grounding_metrics=grounding_metrics,
        confidence=confidence,
        trace=trace
    )
