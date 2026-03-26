from schemas.response import QueryResponse

def build_query_response(
    question,
    pipeline_status,
    limitations,
    answer=None,
    sources=None,
    meta=None,
    confidence=None,
    trace=None
):
    return QueryResponse(
        question=question,
        pipeline_status=pipeline_status,
        answer=answer or [],
        limitations=limitations,
        sources=sources or [],
        meta=meta or {},
        confidence=confidence,
        trace=trace
    )
