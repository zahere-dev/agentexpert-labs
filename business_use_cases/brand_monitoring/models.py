import operator
from typing import Annotated, Optional, TypedDict


# State definition
class AgentState(TypedDict):
    """State that flows through the pipeline"""  
    url: str    
    html_content: Optional[str]
    brand_name: Optional[str]
    themes: Optional[list[str]]
    domain: Optional[str]
    location: Optional[str]
    error: Optional[str]
    queries: Optional[dict[str, list[str]]]
    query_results: Optional[list[dict]]
    analysis_report: Optional[dict]
    
  
    