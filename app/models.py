from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    """Request body for /recommend endpoint"""
    query: str

class Assessment(BaseModel):
    """Individual assessment details"""
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendResponse(BaseModel):
    """Response structure for recommendations"""
    recommended_assessments: List[Assessment]
