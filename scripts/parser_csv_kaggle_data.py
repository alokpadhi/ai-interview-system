import os
import json
import yaml
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from pandas import DataFrame


STOPWORDS = {
    "what", "is", "are", "the", "of", "to", "and", "when", "do", "we",
    "you", "which", "can", "use", "main", "how", "why"
}

DEFAULT_SOURCE = "kaggle_data_science_interviews"

TOPIC_KEYWORDS = {
    "NLP": ["nlp", "token", "embedding", "word2vec", "glove", "fasttext", "language model"],
    "Deep Learning": ["cnn", "gan", "rnn", "lstm", "backprop", "neural"],
    "Machine Learning": ["regression", "classification", "svm", "tree", "bias", "variance"],
    "Data Science": ["pca", "statistics", "data analysis"]
}


def call_ollama(prompt: str, model: str = "llama3.1:8b") -> Dict:
    """Call Ollama and enforce strict JSON output."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True
    )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def detect_csv_schema(df: DataFrame) -> str:
    cols = set(c.lower() for c in df.columns)

    if {"id", "question", "difficulty", "category"}.issubset(cols):
        return "RICH"
    if {"question"}.issubset(cols):
        return "MINIMAL"
    raise ValueError("Unsupported CSV schema")


def normalize_dataframe(df: DataFrame, source: str) -> List[Dict]:
    schema = detect_csv_schema(df)
    normalized = []

    for row in df.itertuples(index=False):
        if schema == "RICH":
            normalized.append({
                "id": str(row.id),
                "question": row.question,
                "difficulty": row.difficulty.lower() if row.difficulty else None,
                "topic": row.category,
                "date": getattr(row, "date", None),
                "source": source
            })
        else:  # MINIMAL
            normalized.append({
                "id": None,
                "question": row.question,
                "difficulty": None,
                "topic": None,
                "date": None,
                "source": source
            })

    return normalized


class CollectCsvInterviewQuestions:
    """
    Heuristics + LLM enrichment
    """

    def infer_topic(self, question: str) -> Optional[str]:
        q = question.lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(k in q for k in keywords):
                return topic
        return None

    def infer_question_type(self, question: str) -> str:
        q = question.lower()

        if q.startswith(("explain", "what is", "define")):
            return "conceptual"
        if q.startswith(("compare", "difference")):
            return "comparative"
        if any(k in q for k in ["implement", "code", "write"]):
            return "coding"
        if any(k in q for k in ["design", "how would you"]):
            return "scenario"

        return "conceptual"

    def infer_difficulty(self, question: str) -> str:
        q = question.lower()

        if any(k in q for k in ["gan", "transformer", "attention"]):
            return "hard"
        if any(k in q for k in ["cnn", "backprop", "word2vec", "gradient"]):
            return "medium"
        return "easy"

    def estimate_time(self, difficulty: str, q_type: str) -> int:
        base = {"easy": 3, "medium": 5, "hard": 8}.get(difficulty, 5)
        if q_type == "coding":
            base += 5
        if q_type == "scenario":
            base += 2
        return base

    def generate_tags(self, question: str, topic: Optional[str]) -> List[str]:
        words = re.findall(r"[a-zA-Z_]+", question.lower())
        keywords = [w for w in words if w not in STOPWORDS and len(w) > 3]
        tags = set(keywords[:6])
        if topic:
            tags.add(topic.lower().replace(" ", "_"))
        return list(tags)


    def llm_enrich(self, question: str) -> Dict:
        prompt = f"""
You are an expert AI interviewer.

Return STRICT JSON only.

Question:
"{question}"

Schema:
{{
  "difficulty": "easy|medium|hard",
  "topic": "string",
  "question_type": "conceptual|coding|scenario|comparative",
  "tags": ["string"],
  "key_concepts": ["string"],
  "reference_answer": "string"
}}
"""
        return call_ollama(prompt)


    def parse(self, row: Dict) -> Dict:
        question = row["question"]

        # Heuristic layer
        heuristic_difficulty = row["difficulty"] or self.infer_difficulty(question)
        heuristic_topic = row["topic"] or self.infer_topic(question)
        question_type = self.infer_question_type(question)
        heuristic_tags = self.generate_tags(question, heuristic_topic)

        # Mandatory LLM layer
        llm_data = self.llm_enrich(question)

        final_difficulty = (
            row["difficulty"]
            or heuristic_difficulty
            or llm_data.get("difficulty", "medium")
        )

        final_topic = (
            row["topic"]
            or heuristic_topic
            or llm_data.get("topic", "General")
        )

        return {
            "text": question,
            "difficulty": final_difficulty,
            "topic": final_topic,
            "question_type": question_type,
            "estimated_time_minutes": self.estimate_time(
                final_difficulty, question_type
            ),
            "tags": list(set(heuristic_tags) | set(llm_data.get("tags", []))),
            "reference_answer": llm_data.get("reference_answer", ""),
            "key_concepts": llm_data.get("key_concepts", []),
            "source": row["source"]
        }


def process_csv(csv_path: str, collector: CollectCsvInterviewQuestions) -> List[Dict]:
    df = pd.read_csv(csv_path)
    normalized = normalize_dataframe(df, DEFAULT_SOURCE)
    parsed_data = []
    for idx, row in enumerate(normalized):
        parsed_data.append(collector.parse(row))
        print(f"Processed rows are: {idx+1}/{len(normalized)}")
    
    return parsed_data


def main():
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config" / "data_sources.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    files = config["data_sources"]["interview_questions"]["raw"][0]["files"]
    processed_path = Path(config["data_sources"]["interview_questions"]["processed_path"])
    processed_path.mkdir(parents=True, exist_ok=True)

    collector = CollectCsvInterviewQuestions()
    all_questions = []

    base_data_path = base_dir / "data/datasets/raw" / "interview_questions"

    for file in files:
        print(f"Started working on {file}")
        all_questions.extend(process_csv(base_data_path / file, collector))

    for idx, q in enumerate(all_questions):
        q["id"] = f"github_ds_interviews_{idx:04d}"

    output_file = processed_path / "kaggle_datascience_interview_questions.json"
    with open(output_file, "w") as f:
        json.dump(all_questions, f, indent=2)

    print(f"Saved {len(all_questions)} questions → {output_file}")


if __name__ == "__main__":
    main()
