"""
Embedding service for text vectorization.
Uses BGE-base-en-v1.5 with normalization for cosine similarity.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
from functools import lru_cache
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Singleton service for text embeddings.
    """
    
    _instance: Optional['EmbeddingService'] = None
    
    def __new__(cls, model_name: str = "BAAI/bge-base-en-v1.5"):
        """Singleton pattern implementation"""
        # TODO: Implement singleton logic
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        """Initialize embedding model (only runs once due to singleton)"""
        # Guard against re-initialization
        if hasattr(self, '_initialized'):
            return
        
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.model.encode("warmup", normalize_embeddings=True)

        self._initialized = True
        
        logger.info(f"Embedding service initialized: {model_name}, dim={self.dimension}")
    
    def embed(self, text: str, normalize: bool = True) -> List[float]:
        """
        Embed single text string.
        
        Args:
            text: Input text to embed
            normalize: Whether to normalize output (default True for BGE)
            
        Returns:
            List of floats (768 dimensions for BGE-base)
        """
        return self.model.encode(text, normalize_embeddings=normalize).tolist()
    
    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding (default 32)
            show_progress: Show progress bar (useful for large batches)
            normalize: Whether to normalize outputs
            
        Returns:
            List of embedding vectors
        """
        return self.model.encode(sentences=texts,
                                 batch_size=batch_size,
                                 show_progress_bar=show_progress,
                                 normalize_embeddings=normalize).tolist()
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        Useful for testing and debugging.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between -1 and 1 (higher = more similar)
        """
        emb1 = np.asarray(self.embed(text=text1), dtype=np.float32)
        emb2 = np.asarray(self.embed(text=text2), dtype=np.float32)
        eps: float = 1e-8
        
        dot = np.dot(emb1, emb2)
        
        denom = max(np.linalg.norm(emb1) * np.linalg.norm(emb2), eps)

        return float(dot / denom)
    
    @property
    def model_name(self) -> str:
        """Get the loaded model name"""
        return self.model.model_name if hasattr(self, 'model') else None


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """
    Get singleton embedding service instance.
    """
    return EmbeddingService()