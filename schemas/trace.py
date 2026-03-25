from pydantic import BaseModel
from typing import List, Dict, Optional


class AnalysisTrace(BaseModel):
    query_expansion: Optional[List] = None
    chunks_provided_to_synthesizer: Optional[List[Dict]] = None
    paper_stats: Optional[List[Dict]] = None 