"""
Data ingestion pipeline for ChromaDB.
Loads processed data and ingests into vector store.
"""

import json
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm

from src.data.vector_store import VectorStore
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles ingestion of all data into ChromaDB"""
    
    def __init__(self, vector_store: VectorStore, reset_collections: bool = False):
        """
        Args:
            vector_store: VectorStore instance
            reset_collections: If True, delete existing collections before ingesting
        """
        self.vector_store = vector_store
        self.reset_collections = reset_collections
    
    def load_json(self, file_path: str) -> List[Dict]:
        """
        Load and validate JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of dictionaries
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"{file_path} doesn't exist.")
        
        try:
            with open(file_path, "r") as fp:
                data = json.load(fp)
                return data
        except Exception as e:
            raise ValueError(f"json file couldn't be loaded. {e}")


    def ingest_questions(self, questions_file: str) -> int:
        """
        Ingest interview questions into ChromaDB.
        
        Args:
            questions_file: Path to questions JSON
            
        Returns:
            Number of questions ingested
        """
        logger.info(f"Ingesting questions from {questions_file}")
        
        data = self.load_json(questions_file)
        if not data:
            logger.warning(f"No data found in {questions_file}")
            return 0
            
        collection_name = "interview_questions"
        self.vector_store.create_collection(collection_name, reset=self.reset_collections)
        
        documents = []
        metadatas = []
        ids = []
        
        for item in tqdm(data, desc="Processing Questions"):
            documents.append(item.get("text", ""))
            ids.append(item.get("id"))
            
            meta = item.copy()
            for key, value in meta.items():
                if isinstance(value, (list, dict)):
                    meta[key] = json.dumps(value)
                elif value is None:
                    meta[key] = ""
            metadatas.append(meta)
            
        if documents:
            self.vector_store.add_documents(
                collection_name=collection_name,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
        logger.info(f"Successfully ingested {len(documents)} questions into '{collection_name}'")
        return len(documents)
        
    
    def ingest_concepts(self, concepts_file: str) -> int:
        """
        Ingest ML concepts into ChromaDB.
        
        Args:
            concepts_file: Path to concepts JSON
            
        Returns:
            Number of concepts ingested
        """
        logger.info(f"Ingesting concepts from {concepts_file}")
        
        data = self.load_json(concepts_file)
        if not data:
            logger.warning(f"No data found in {concepts_file}")
            return 0
            
        collection_name = "ml_concepts"
        self.vector_store.create_collection(collection_name, reset=self.reset_collections)
        
        documents = []
        metadatas = []
        ids = []
        
        for item in tqdm(data, desc="Processing Concepts"):
            name = item.get("concept_name", "")
            explanation = item.get("explanation", "")
            doc_text = f"{name}: {explanation}"
            documents.append(doc_text)
            ids.append(item.get("id"))
            
            meta = item.copy()
            for key, value in meta.items():
                if isinstance(value, (list, dict)):
                    meta[key] = json.dumps(value)
                elif value is None:
                    meta[key] = ""
            metadatas.append(meta)
            
        if documents:
            self.vector_store.add_documents(
                collection_name=collection_name,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
        logger.info(f"Successfully ingested {len(documents)} concepts into '{collection_name}'")
        return len(documents)
    
    def ingest_code_solutions(self, solutions_file: str) -> int:
        """
        Ingest code solutions into ChromaDB.
        
        Args:
            solutions_file: Path to solutions JSON
            
        Returns:
            Number of solutions ingested
        """
        logger.info(f"Ingesting code solutions from {solutions_file}")
        
        data = self.load_json(solutions_file)
        if not data:
            logger.warning(f"No data found in {solutions_file}")
            return 0
            
        collection_name = "code_solutions"
        self.vector_store.create_collection(collection_name, reset=self.reset_collections)
        
        documents = []
        metadatas = []
        ids = []
        
        for item in tqdm(data, desc="Processing Solutions"):
            prob_desc = item.get("problem_description", "")
            meaning = item.get("explanation", "")
            doc_text = f"Problem: {prob_desc}\n\nExplanation: {meaning}"
            documents.append(doc_text)
            ids.append(item.get("id"))
            
            meta = item.copy()
            for key, value in meta.items():
                if isinstance(value, (list, dict)):
                    meta[key] = json.dumps(value)
                elif value is None:
                    meta[key] = ""
            metadatas.append(meta)
            
        if documents:
            self.vector_store.add_documents(
                collection_name=collection_name,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
        logger.info(f"Successfully ingested {len(documents)} solutions into '{collection_name}'")
        return len(documents)
    
    def verify_ingestion(self, collection_name: str, expected_count: int) -> bool:
        """
        Verify collection has expected number of documents.
        
        Args:
            collection_name: Collection to verify
            expected_count: Expected document count
            
        Returns:
            True if count matches
        """
        try:
            stats = self.vector_store.get_collection_stats(collection_name)
            actual_count = stats.get("count", 0)
            
            matches = actual_count == expected_count
            status = "VALID" if matches else "INVALID"
            logger.info(f"Verification {status} for {collection_name}: Expected={expected_count}, Actual={actual_count}")
            
            return matches
        except Exception as e:
            logger.error(f"Verification failed for {collection_name}: {e}")
            return False


def main():
    """Run full ingestion pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest data into ChromaDB")
    parser.add_argument('--reset', action='store_true', help='Reset collections')
    parser.add_argument(
        '--questions',
        type=str,
        default='data/datasets/processed/interview_questions/final_interview_questions.json'
    )
    parser.add_argument(
        '--concepts',
        type=str,
        default='data/datasets/processed/concepts/final_concepts.json'
    )
    parser.add_argument(
        '--solutions',
        type=str,
        default='data/datasets/processed/solutions/leetcode_solutions.json'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize
    settings = get_settings()
    vector_store = VectorStore(persist_directory=str(settings.vector_db_path))
    ingestion = DataIngestion(vector_store, reset_collections=args.reset)
    
    # Execute Ingestion
    print("\n=== Starting Ingestion Pipeline ===\n")
    
    # 1. Ingest questions
    q_count = ingestion.ingest_questions(args.questions)
    ingestion.verify_ingestion("interview_questions", q_count)
    
    # 2. Ingest concepts
    c_count = ingestion.ingest_concepts(args.concepts)
    ingestion.verify_ingestion("ml_concepts", c_count)
    
    # 3. Ingest code solutions
    s_count = ingestion.ingest_code_solutions(args.solutions)
    ingestion.verify_ingestion("code_solutions", s_count)
    
    print("\n=== Ingestion Complete ===")
    print(f"Total Questions: {q_count}")
    print(f"Total Concepts:  {c_count}")
    print(f"Total Solutions: {s_count}")


if __name__ == "__main__":
    main()