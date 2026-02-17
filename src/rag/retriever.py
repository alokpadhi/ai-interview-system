"""
High level retrieval interface for RAG operations.
Abstracts chromadb operations with agent-fiendly API.
"""

from typing import List, Optional, Dict, Set
from src.utils.logging_config import get_logger
from src.data.vector_store import VectorStore
from src.rag.models import RetrievalResult, RetrievalContext

logger = get_logger(__name__)

class VectorRetriever:
    """High level interface for semantic retrieval.
    """
    def __init__(self, vector_store: VectorStore):
        """
        Args:
            vector_store (VectorStore): VectorStore instance (chromadb wrapper)
        """
        self.vector_store = vector_store
        logger.info("VectorRetriever initialized")

    def retrieve_questions(
            self,
            query: str,
            difficulty: Optional[str] = None,
            topic: Optional[str] = None,
            question_type: Optional[str] = None,
            exclude_ids: Optional[Set[str]] = None,
            n_result: int = 10
    ) -> List[RetrievalResult]:
        """Retrieve interview questions with optional filters.

        Args:
            query (str): Text query for semantic search.
            difficulty (Optional[str], optional): Filter by difficulty (easy, medium, hard). Defaults to None.
            topic (Optional[str], optional): Filter by topic. Defaults to None.
            question_type (Optional[str], optional): Filter by type. Defaults to None.
            exclude_ids (Optional[Set[str]], optional): Set of question IDs to exclude (already asked). Defaults to None.
            n_result (int, optional): Number of results to return. Defaults to 10.

        Returns:
            List[RetrievalResult]: List of RetrievalResult objects, sorted by relevance.
        
        Example:
            results = retriever.retrieve_questions(
                query="optimization algorithms",
                difficulty="medium",
                exclude_ids={"q_001", "q_002"},
                n_results=5
            )
        """
        logger.info(
            f"Retrieving questions: query='{query}', difficulty={difficulty}, "
            f"topic={topic}, type={question_type}, exclude={len(exclude_ids or set())} ids"
        )
        
        try:
            # 1. Build where clause using _build_where_clause()
            where_clause = self._build_where_clause(
                difficulty=difficulty,
                topic=topic,
                question_type=question_type
            )
            
            # 2. Call self.vector_store.query() with collection="interview_questions"
            raw_results = self.vector_store.query(
                collection_name="interview_questions",
                query_text=query,
                n_results=n_result,
                where=where_clause
            )
            
            # 3. Format results using _format_results()
            # 4. Filter out exclude_ids (already done in _format_results)
            results = self._format_results(raw_results, exclude_ids=exclude_ids)
            
            # 5. Log result count
            logger.info(f"Retrieved {len(results)} questions (after exclusions)")
            
            # 6. Return results
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve questions: {e}")
            # Handle empty results gracefully - return empty list on error
            return []

    def retrieve_concepts(
        self,
        query: str,
        category: Optional[str] = None,
        n_results: int = 3
    ) -> List[RetrievalResult]:
        """
        Retrieve ML concepts for explanations.
        
        Used by Feedback Agent when candidate needs concept clarification.
        
        Args:
            query: Concept to look up (e.g., "gradient descent", "overfitting")
            category: Filter by category (optimization, evaluation, etc.)
            n_results: Number of concepts to return (default 3)
            
        Returns:
            List of RetrievalResult objects
            
        Example:
            concepts = retriever.retrieve_concepts(
                query="regularization techniques",
                category="evaluation",
                n_results=2
            )
        """
        logger.info(f"Retrieving concepts: query='{query}', category={category}")
        
        try:
            # Build where clause for category filter
            where_clause = self._build_where_clause(category=category)
            
            # Query the ml_concepts collection
            raw_results = self.vector_store.query(
                collection_name="ml_concepts",
                query_text=query,
                n_results=n_results,
                where=where_clause
            )
            
            # Format results (no exclude_ids for concepts)
            results = self._format_results(raw_results)
            
            logger.info(f"Retrieved {len(results)} concepts")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve concepts: {e}")
            # Return empty list on error rather than crashing
            return []
    
    def retrieve_code_solutions(
        self,
        query: str,
        language: Optional[str] = None,
        n_results: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve code solutions.
        
        Phase 2 feature - minimal implementation for now.
        
        Args:
            query: Problem description
            language: Programming language filter (python, javascript, etc.)
            n_results: Number of solutions
            
        Returns:
            List of RetrievalResult objects
        """
        logger.info(f"Retrieving code solutions: query='{query}', language={language}")
        
        try:
            # Build where clause for language filter
            where_clause = self._build_where_clause(language=language)
            
            # Query the code_solutions collection
            raw_results = self.vector_store.query(
                collection_name="code_solutions",
                query_text=query,
                n_results=n_results,
                where=where_clause
            )
            
            # Format results (no exclude_ids for code solutions)
            results = self._format_results(raw_results)
            
            logger.info(f"Retrieved {len(results)} code solutions")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve code solutions: {e}")
            # Return empty list on error rather than crashing
            return []
    
    def _build_where_clause(
        self,
        difficulty: Optional[str] = None,
        topic: Optional[str] = None,
        question_type: Optional[str] = None,
        category: Optional[str] = None,
        language: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Build ChromaDB where clause from filters.
        
        ChromaDB where clause format:
        {"field1": "value1", "field2": "value2"}
        
        Only include non-None filters.
        
        Args:
            Various filter parameters
            
        Returns:
            Dictionary for ChromaDB where clause, or None if no filters
            
        Example:
            _build_where_clause(difficulty="medium", topic="optimization")
            → {"difficulty": "medium", "topic": "optimization"}
        """
        where_clause = {
            "difficulty": difficulty,
            "topic": topic,
            "question_type": question_type,
            "category": category,
            "language": language
        }

        where_clause = {k:v for k, v in where_clause.items() if v is not None}

        return where_clause or None
    
    def _format_results(
        self,
        raw_results: Dict,
        exclude_ids: Optional[Set[str]] = None
    ) -> List[RetrievalResult]:
        """
        Convert ChromaDB raw results to RetrievalResult objects.
        
        ChromaDB returns:
        {
            'ids': [['id1', 'id2', ...]],
            'documents': [['doc1', 'doc2', ...]],
            'metadatas': [[{...}, {...}, ...]],
            'distances': [[0.2, 0.3, ...]]
        }
        
        Args:
            raw_results: Raw results from vector_store.query()
            exclude_ids: IDs to filter out
            
        Returns:
            List of RetrievalResult objects, sorted by relevance (highest first)
        """
        if not raw_results or not raw_results.get('ids') or len(raw_results['ids']) == 0:
            logger.debug("No results returned from vector store")
            return []
        
        # ChromaDB returns nested lists, extract the first element
        ids = raw_results['ids'][0]
        documents = raw_results['documents'][0]
        metadatas = raw_results.get('metadatas', [[]])[0]
        distances = raw_results.get('distances', [[]])[0]
        
        # Check if the first nested list is empty
        if len(ids) == 0:
            logger.debug("Empty result set from vector store")
            return []
        
        exclude_ids = exclude_ids or set()
        results = []
        
        # Iterate through results and create RetrievalResult objects
        for i, doc_id in enumerate(ids):
            # Skip excluded IDs
            if doc_id in exclude_ids:
                logger.debug(f"Skipping excluded ID: {doc_id}")
                continue
            
            # Get document text
            text = documents[i] if i < len(documents) else ""
            
            # Get metadata (default to empty dict if missing)
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # Convert distance to relevance score
            # ChromaDB uses L2 distance (lower is better)
            # Convert to similarity score (higher is better) in range [0, 1]
            distance = distances[i] if i < len(distances) else 1.0
            
            # Clamp distance to valid range and convert to relevance
            # For L2 distance, we use: relevance = 1.0 / (1.0 + distance)
            # This ensures relevance is in [0, 1] and higher for smaller distances
            try:
                distance = max(0.0, float(distance))
                relevance_score = 1.0 / (1.0 + distance)
                # Clamp to [0, 1] range
                relevance_score = max(0.0, min(1.0, relevance_score))
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid distance value for ID {doc_id}: {distance}, using 0.0. Error: {e}")
                relevance_score = 0.0
            
            # Create RetrievalResult object
            try:
                result = RetrievalResult(
                    id=doc_id,
                    text=text,
                    relevance_score=relevance_score,
                    metadata=metadata
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to create RetrievalResult for ID {doc_id}: {e}")
                continue
        
        # Sort by relevance_score descending (highest relevance first)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.debug(f"Formatted {len(results)} results (excluded {len(exclude_ids)} IDs)")
        return results
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """
        Get statistics about a collection.
        
        Useful for debugging and system monitoring.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            Dictionary with stats (name, count, metadata)
        """
        logger.info(f"Getting stats for collection: {collection_name}")
        try:
            stats = self.vector_store.get_collection_stats(collection_name)
            logger.debug(f"Collection '{collection_name}' stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats for collection '{collection_name}': {e}")
            raise
    
    def get_all_collections(self) -> List[str]:
        """Get list of all collection names"""
        logger.info("Listing all collections")
        try:
            collections = self.vector_store.list_collections()
            logger.debug(f"Found {len(collections)} collections: {collections}")
            return collections
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise
