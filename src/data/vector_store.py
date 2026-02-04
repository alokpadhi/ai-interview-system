from typing import List, Dict, Any, Optional, TYPE_CHECKING
import os
from pathlib import Path
import chromadb
from chromadb import Collection
from chromadb.utils import embedding_functions
from src.utils.logging_config import get_logger
import torch

if TYPE_CHECKING:
    from chromadb.api import ClientAPI

logger = get_logger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "data/vector_db"):
        """initialize the vector store
        """
        logger.info("ChromaDB initializing...")
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client: "ClientAPI" = chromadb.PersistentClient(
            persist_directory
        )
        logger.info("ChromaDB initialized at {persistent_directory}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-base-en-v1.5",
            device=device,
            normalize_embeddings=True
        )

    def create_collection(self, 
                          collection_name: str, 
                          metadata: Optional[Dict] = {
                              "hnsw:space": "cosine",
                              "hnsw:M": 32,  # Increase connections (default 16)\
                              "hnsw:ef_construction": 200,  # Better index quality
                              "hnsw:ef_search": 100  # Better search quality
                              }, 
                          reset: bool = False) -> Collection:
        """Create or delete collection in chromadb.

        Args:
            collection_name (_type_, optional): collection name. Defaults to str.
            metadata (Optional[Dict], optional): meta data. Defaults to None.
            reset (bool, optional): Delete the collection Flag. Defaults to False.
        """
        if reset:
            self.client.delete_collection(name=collection_name)
            logger.info(f"{collection_name} collection is deleted")

        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata=metadata
        )
        logger.info(f"collection created with name: {collection.name}")
        return collection

    def modify_collection(self,
                          collection: Collection,
                          collection_name: str,
                          metadata: Optional[Dict]=None) -> Collection:
        collection = collection.modify(name=collection_name, metadata=metadata)

        return collection

    def add_documents():
        #TODO
        pass

    def query():
        #TODO
        pass

    def get_collection_stats():
        #TODO
        pass

    def list_collections():
        #TODO
        pass


