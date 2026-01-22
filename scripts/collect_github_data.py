import yaml
import os
from pathlib import Path
import re
import requests
from typing import List, Dict
import json


STOPWORDS = {
    "what", "is", "are", "the", "of", "to", "and", "when", "do", "we",
    "you", "which", "can", "use", "main"
}

class CollectGithubInterviewQuestions:
    def __init__(self, github_url=None):
        self.github_url = github_url
    
    def get_response(self):
        if self.github_url:
            response = requests.get(self.github_url)
            response.raise_for_status()
            content = response.text

        return content
    
    def _normalize_question(self, question: str):
        difficulty = "medium"
        if "👶" in question:
            difficulty = "easy"
        elif "⭐" in question:
            difficulty = "hard"

        clean_question = re.sub(r"[👶⭐]", "", question).strip()
        return clean_question, difficulty
    
    def _infer_question_type(self, question: str, answer: str) -> str:
        text = question.lower() + ' ' + answer.lower()

        # if any(k in text for k in ["code", "program", "function", "python", "sql"]):
        #     return "coding"
        if any(k in text for k in ["derive", "prove", "formula", "equation"]):
            return "math"
        if any(k in text for k in ["how would you", "what would you do", "design", "would you"]):
            return "scenario"

        return "conceptual"
    
    def _estimate_time(self, q_type: str, difficulty: str) -> int:
        base_times = {
            'conceptual': 3,
            'coding': 10,
            'scenario': 7,
            'math': 5
        }
        
        multipliers = {
            'easy': 0.8,
            'medium': 1.0,
            'hard': 1.5
        }
        
        base = base_times.get(q_type, 5)
        multiplier = multipliers.get(difficulty, 1.0)
        
        return int(base * multiplier)

    def _generate_tags(self, question: str, topic: str) -> List[str]:
        words = re.findall(r"[a-zA-Z_]+", question.lower())
        keywords = [
            w for w in words
            if w not in STOPWORDS and len(w) > 3
        ]

        return list(set([topic] + keywords[:5]))
    
    def _extract_key_concepts(self, answer: str) -> List[str]:
        bold_terms = re.findall(r"\*\*(.*?)\*\*", answer)
        italic_terms = re.findall(r"\*(.*?)\*", answer)

        candidates = set(bold_terms + italic_terms)

        cleaned = [
            c.strip().lower().replace(" ", "_")
            for c in candidates
            if len(c) > 2
        ]

        return list(set(cleaned))
    
    def _infer_difficulty(self, question: str, answer: str, difficulty: str) -> str:
        answer_length = len(answer.split())

        advanced_terms = [
            'theorem', 'proof', 'derive', 'mathematical', 'formally',
            'advanced', 'complex', 'sophisticated', 'intricate'
        ]

        text_lower = (question + ' ' + answer).lower()

        has_advanced = any(term in text_lower for term in advanced_terms)

        if difficulty == "hard":
            if has_advanced or answer_length > 200:
                return "hard"
            else:
                return "medium"
        elif difficulty == "easy":
            return "easy"
        else:
            return "NA"
        
    def _create_question_dict(self, 
                             raw_question: str, 
                             answer: str, 
                             topic: str) -> Dict:
        
        # Infer metadata
        question, difficulty = self._normalize_question(raw_question)
        difficulty = self._infer_difficulty(question, answer, difficulty)
        question_type = self._infer_question_type(question, answer)
        tags = self._generate_tags(question, topic)
        concepts = self._extract_key_concepts(answer)


        
        return {
            'text': question,
            'difficulty': difficulty,
            'topic': topic,
            'question_type': question_type,
            'estimated_time_minutes': self._estimate_time(question_type, difficulty),
            'tags': tags,
            'reference_answer': answer,
            'key_concepts': concepts,
            'source': 'github_data_science_interviews'
        }
    
    def parse_github_markdown(self, content: str) -> List[Dict]:
        SECTION_PATTERN = r'(?m)^##\s(.*)\n([\s\S]*?)(?=^##|\Z)'
        QA_PATTERN = r'\*\*(.*?)\*\*\n\n([\s\S]*?)(?=\n<br/>\n|\Z)'

        interview_questions = []
        for section_title, section_body in re.findall(SECTION_PATTERN, content):
            topic = section_title.strip().lower().replace(" ", "_")

            for raw_question, answer in re.findall(QA_PATTERN, section_body):
                interview_question_dict = self._create_question_dict(raw_question=raw_question,
                                                                     answer=answer,
                                                                     topic=topic)
                
                interview_questions.append(interview_question_dict)
        return interview_questions

def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_source_config_path = base_dir / "config" / "data_sources.yaml"
    
    if not os.path.exists(data_source_config_path):
        raise FileNotFoundError(f"Config file not found at {data_source_config_path}")

    with open(data_source_config_path, "r") as f:
        print("Reading yaml file")
        config_dict = yaml.safe_load(f)

    urls = config_dict["data_sources"]["interview_questions"]["raw"][1]["urls"]
    processed_path = config_dict["data_sources"]["interview_questions"]["processed_path"]

    all_interview_questions = []
    print("Parsing interview questions...")
    for idx, url in enumerate(urls):
        interview_question_collector = CollectGithubInterviewQuestions(url)
        print(f"Fetching markdown file from url#{idx+1}...")
        content = interview_question_collector.get_response()
        print(f"Parsing markdown file from url#{idx+1}...")
        interview_questions = interview_question_collector.parse_github_markdown(content)
        print(f"Parsed questions with {len(interview_questions)} number of question.")
        all_interview_questions.extend(interview_questions)
        
    
    print(f"Total question parses is {len(all_interview_questions)}")

    for idx, question in enumerate(all_interview_questions):
        question["id"] = f"github_ds_interviews_{idx:04d}"

    # Display sample
    print("\n" + "="*60)
    print("SAMPLE QUESTIONS")
    print("="*60)
    for q in all_interview_questions[:3]:
        print(q)

    output_file = Path(processed_path) / "github_datascience_interview_questions.json"

    with open(output_file, "w") as fp:
        json.dump(all_interview_questions, fp, indent=2)

    print(f"\nSaved {len(all_interview_questions)} questions to {output_file}")

if __name__ == "__main__":
    main()
    






        

