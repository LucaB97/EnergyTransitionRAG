from pydantic import BaseModel, Field
from typing import Optional


class Source(BaseModel):
    paper_id: str = Field(description="Unique identifier of the source paper")
    title: str = Field(description="Title of the academic paper")
    authors: str = Field(description="Authors of the paper")
    year: int = Field(description="Publication year")
    journal: Optional[str] = Field(
        None,
        description="Journal or conference where the paper was published"
    )
    citation_number: Optional[int] = Field(
        None,
        description="Numeric citation identifier used in the answer"
    )