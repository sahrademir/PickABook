from pydantic import BaseModel
from typing import List, Optional

# Request Schemas

class ContentBasedRequest(BaseModel):
    book_title: str
    top_n: Optional[int] = 5

class SemanticSearchRequest(BaseModel):
    query_text: str
    top_n: Optional[int] = 5

class UserRecommendationRequest(BaseModel):
    favorite_books: List[str]
    top_n: Optional[int] = 5

# Response Schemas

class BookRecommendationResult(BaseModel):
    title: str
    isbn: Optional[str] = None
    similarity_score: float
    llm_explanation: Optional[str] = None

class RecommendationResponse(BaseModel):
    status: str
    method: str
    results: List[BookRecommendationResult]