from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union


class AxisProfile(BaseModel):
    level: Optional[str] 
    # Literal["Strong", "Moderate", "Weak", "Not_applicable"]
    score: Optional[float] = Field(ge=0.0, le=1.0)
    explanation: Optional[
        Union[str, List[str]]
    ] = None

class ConfidenceProfile(BaseModel):
    evidence: AxisProfile = Field(
        default_factory=lambda: AxisProfile(level="Not_applicable", score=None),
        description="Judgement of relevance of the evidence to the query"
    )
    grounding: AxisProfile = Field(
        default_factory=lambda: AxisProfile(level="Not_applicable", score=None),
        description="Grounding quality of the synthesis"
    )
    status: Literal["Success", "Not applicable"]
    reason: Optional[str] = Field(
        default="",
        description="Reason for \"Not applicable\" status"
    )

class GroundingMetrics(BaseModel):
    available_chunks: int
    used_chunks: int
    chunk_coverage: float

    available_papers: int
    used_papers: int
    paper_dominance: float

    avg_citations_per_sentence: float
    multi_source_sentence_ratio: float