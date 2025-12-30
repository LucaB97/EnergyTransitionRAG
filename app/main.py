from fastapi import FastAPI
from fastapi import Request

from app.dependencies import load_system
from app.schemas import QueryRequest, QueryResponse, RetrievedChunk


app = FastAPI(
    title="Environmental Research Synthesizer",
    description="Semantic retrieval + evidence-based synthesis from academic literature",
    version="0.1.0"
)


@app.on_event("startup")
def startup_event():
    load_system(app)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest, req: Request):
    retriever = req.app.state.retriever
    synthesizer = req.app.state.synthesizer

    results = retriever.search(
        request.question,
        top_k=request.top_k
    )

    answer = synthesizer.synthesize(
        request.question,
        results
    )

    retrieved_chunks = [
        RetrievedChunk(
            paper_id=c["paper_id"],
            title=c["title"],
            authors=c["authors"],
            year=c["year"],
            text=c["text"]
        )
        for c in results
    ]

    return QueryResponse(
        question=request.question,
        retrieved_chunks=retrieved_chunks,
        answer=answer
    )
