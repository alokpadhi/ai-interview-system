"""
Data models for RAG operations.
Using pydantic for validation and type safety.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any


class RetrievalResult(BaseModel):
    """Single retrieval result with metadata.
    """
    id: str
    text: str = Field(..., description="The question/concept text to embed")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="cosine similarity (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("relevance_score")
    @classmethod
    def check_relevance_score(cls, score):
        if score >=0.0 and score <= 1.0:
            return score
        raise ValueError("Relevance score should be in between 0 and 1.")
    
    @property
    def difficulty(self) -> Optional[str]:
        """Get difficulty from metadata"""
        return self.metadata.get("difficulty")
    
    @property
    def topic(self) -> Optional[str]:
        """Get topic from metadata"""
        return self.metadata.get("topic")
    
    @property
    def question_type(self) -> Optional[str]:
        """Get question_type from metadata (for questions)"""
        return self.metadata.get("question_type")
    
    @property
    def concept_name(self) -> Optional[str]:
        """Get concept_name from metadata (for concepts)"""
        return self.metadata.get("concept_name")
    
    def __str__(self) -> str:
        """Human readable representation"""
        return f"[{self.relevance_score:.3f}] {self.text}"
    
    def __repr__(self) -> str:
        """Developer-friendly representation"""
        return f"RetrievalResult(id={self.id!r}, score={self.relevance_score:.3f})"
    

class RetrievalContext(BaseModel):
    """
    Context for retrieval decisions.
    Used by self-rag to decide whether retrieve or not.
    """
    # Interview state
    topics_covered: List[str] = Field(default_factory=list)
    difficulty_level: Optional[str] = None
    last_question_id: Optional[str] = None
    last_question_topic: Optional[str] = None

    # Performance tracking
    performance_trajectory: List[float] = Field(default_factory=list)
    question_count: int = 0
    average_score: float = 0.0

    # Retrieval history (for self RAG)
    last_retrieval_time: Optional[float] = None
    cached_candidates_available: bool = False

    def should_adapt_difficulty(self) -> bool:
        """Heruristic: Should we adjust difficulty based on performance?"""
        if len(self.performance_trajectory) < 3:
            return False

        last3 = self.performance_trajectory[-3:]

        high = all(score >= 8.0 for score in last3)
        low = all(score <= 5.0 for score in last3)

        return high or low
    
    def get_performance_trend(self) -> str:
        """Return the performance trend from ['improving', 'declining', 'stable']"""
        ### (x-x_mean): how far each time point is from avg time
        ### (y-y_mean): how far each score is from the average score
        ### if x, y move in same direction the (x-xmean)*(y-ymean) is positive
        ### if x,y move in opposite direction then the product is negative
        ### sum((x-xmean)*(y-ymean)): covariance (how much x and y move together)
        ### (x-xmean)**2: how much x moves
        ### slope = change in y/change in x -> cov(x,y)/var(x)
        n = len(self.performance_trajectory)
        if n < 2:
            return "stable" # not enough data
        
        x = list(range(n))
        y = self.performance_trajectory

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((xi - x_mean) * (yi-y_mean) for xi, yi in zip(x,y))
        denominator = sum((xi-x_mean)**2 for xi in x)

        slope = numerator / denominator if denominator != 0 else 0.0

        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"