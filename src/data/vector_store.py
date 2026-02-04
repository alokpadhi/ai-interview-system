"""
ChromaDB vector store client.
Handles collection management, document insertion, and hybrid search.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Any
from pathlib import Path
from src.utils.config import get_settings
from src.utils.logging_config import get_logger


settings = get_settings()
logger = get_logger(__name__)


class VectorStore:
    """
    ChromaDB client with production-optimized configuration.
    
    Features:
    - Persistent storage
    - HNSW index tuning for 768-dim embeddings
    - Hybrid search (semantic + metadata filtering)
    - Batch operations
    """
    
    def __init__(self, persist_directory: str = "data/vector_db"):
        """
        Initialize ChromaDB client with persistence.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
        """
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )

        logger.info(f"ChromaDB client initialized at {persist_directory}.")
    
    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict] = None,
        reset: bool = False
    ):
        """
        Create or get collection with HNSW optimization.
        
        Args:
            name: Collection name
            metadata: Optional collection metadata
            reset: If True, delete existing collection first
            
        Returns:
            Collection object
        """
        if reset:
            try:
                self.delete_collection(name)
            except Exception:
                pass  # Ignore if it doesn't exist
        hnsw_config = {
            "hnsw:space": "cosine",
            "hnsw:M": 32,
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 100
        }
        collection_metadata = {**hnsw_config, **(metadata or {})}
        collection = self.client.get_or_create_collection(
            name=name,
            metadata=collection_metadata
        )
        logger.info(f"Collection {collection.name} is created.")
        return collection

    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str],
        batch_size: int = 1000
    ):
        """
        Add documents to collection in batches.
        
        Args:
            collection_name: Target collection
            documents: List of text documents to embed
            metadatas: List of metadata dicts
            ids: List of unique IDs
            batch_size: Batch size for insertion (ChromaDB recommends 1000)
        """
        if not (len(documents) == len(metadatas) == len(ids)):
            raise ValueError(
                f"Length mismatch: documents={len(documents)},"
                f"metadatas={len(metadatas)}, ids={len(ids)}"
            )
        
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs detected. All document IDs must be unique.")
        
        if not documents:
            return 
        
        collection = self.client.get_collection(collection_name)
        total_docs = len(documents)

        logger.info(f"Starting batch insert into '{collection_name}' | total_docs={total_docs}, batch_size={batch_size}")

        for start in range(0, total_docs, batch_size):
            end = min(start + batch_size, total_docs)

            batch_docs = documents[start:end]
            batch_ids = ids[start:end]
            batch_metadatas = metadatas[start:end]

            collection.add(
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            logger.info(f"Inserted batch {start}:{end} ({end}/{total_docs})")

        logger.info(f"Completed inserting {total_docs} documents into '{collection_name}'")

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict:
        """
        Query collection with hybrid search (semantic + metadata filters).
        
        Args:
            collection_name: Collection to query
            query_text: Text query for semantic search
            n_results: Number of results to return
            where: Metadata filters (e.g., {"difficulty": "medium"})
            where_document: Document content filters
            
        Returns:
            Dict with keys: ids, documents, metadatas, distances
        """
        if not collection_name:
            raise ValueError(f"Collection doesn't exist. Please create the collection first with create_collection().")
        collection = self.client.get_collection(collection_name)
        results = collection.query(
            query_texts=query_text,
            n_results=n_results,
            where=where,
            where_document=where_document
        )

        return results

    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Dict with count, metadata, etc.
        """
        if not collection_name:
            raise ValueError(f"Collection doesn't exist. Please create the collection first with create_collection()")
        
        collection = self.client.get_collection(collection_name)

        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
    
    def list_collections(self) -> List[str]:
        """List all collection names"""
        cols = self.client.list_collections()
        return [c.name for c in cols]
    
    def delete_collection(self, collection_name: str):
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            if "does not exist" in str(e).lower():
                logger.info(f"Collection {collection_name} does not exist, skipping delete")
            else:
                raise



def get_vector_store() -> VectorStore:
    """Get vector store instance (can be singleton later if needed)"""
    from src.utils.config import get_settings
    settings = get_settings()
    return VectorStore(persist_directory=str(settings.vector_db_path))