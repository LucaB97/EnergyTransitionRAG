from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

from schemas.source import Source
from schemas.confidence import ConfidenceProfile
from schemas.trace import AnalysisTrace 


class Sentence(BaseModel):
    text: str = Field(
        ...,
        description="Concise evidence-based statement grounded in the retrieved literature",
        example="Operational and maintenance costs for wind energy are higher than for coal-fired generation."
    )
    citations: List[str] = Field(
        ...,
        description="List of source citations supporting the claim, formatted as 'Author, Year'"
    )


class QueryResponse(BaseModel):
    question: str = Field(description="Original research question")
    
    pipeline_status: Literal[
        "success",
        "out_of_scope",
        "retrieval_failed",
        "generation_error"
    ] = Field(
        description="Technical execution status of the pipeline"
    )
    
    answer: List[Sentence] = Field(
        description="Structured synthesis of the retrieved evidence"
    )
    
    limitations: List[str] = Field(
        description="Known limitations, uncertainties, or gaps in the available evidence"
    )
    
    sources: List[Source] = Field(
        description="Academic sources that support the synthesized answer"
    )
    
    meta: Dict = Field(
        default_factory=dict,
        description="Additional metadata about retrieval and synthesis"
    )
    
    confidence: Optional[ConfidenceProfile] = Field(
        default=None,
        description="Overall confidence in the synthesized answer; None if pipeline failed"
    )
    
    trace: Optional[AnalysisTrace] = None