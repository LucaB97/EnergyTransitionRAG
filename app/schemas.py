from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class RetrievedChunk(BaseModel):
    paper_id: str
    title: str
    authors: str
    year: int
    text: str


class QueryResponse(BaseModel):
    question: str
    retrieved_chunks: List[RetrievedChunk]
    answer: str
