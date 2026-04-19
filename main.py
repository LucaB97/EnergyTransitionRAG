from fastapi import FastAPI, Request

from dependencies2 import load_system
from schemas.request import QueryRequest
from schemas.response import QueryResponse


app = FastAPI(
    title="Environmental Research Synthesizer",
    description="Semantic retrieval + evidence-based synthesis from academic literature",
    version="0.1.0"
)


@app.on_event("startup")
def startup_event():
    load_system(app)


@app.get("/health")
def health_check(req: Request):
    pipeline = getattr(req.app.state, "pipeline", None)
    retriever = getattr(pipeline, "retriever", None)

    return {
        "status": "ok",
        "rag_pipeline_loaded": pipeline is not None,
        "metadata_loaded": hasattr(pipeline, "metadata"),
        "scope_classifier_loaded": hasattr(pipeline, "scope_classifier"),
        "retriever_loaded": retriever is not None,
        "index_loaded": (
            hasattr(retriever.semantic_retriever, "index")
            and retriever.semantic_retriever.index is not None
        ),
        "index_size": retriever.semantic_retriever.index.ntotal if retriever else None,
        "relevance_profiler_loaded": hasattr(pipeline, "relevance_profiler"),
        "query_expander_loaded": hasattr(pipeline, "query_expander"),
        "synthesizer_loaded": hasattr(pipeline, "synthesizer"),
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest, req: Request):
    pipeline = req.app.state.pipeline
    return pipeline.run(request)