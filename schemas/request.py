from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="Natural language research question to be answered using the indexed literature",
        example="What are the economic impacts of replacing fossil fuels with wind energy?"
    )
    topk_faiss: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Number of chunks to be retrieved based on semantic similarity"
    )
    topk_bm25: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Number of chunks to be retrieved based on lexical matches"
    )