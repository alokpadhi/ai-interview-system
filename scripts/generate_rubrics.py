# scripts/generate_rubrics.py
from pathlib import Path
import json
import copy
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import re


CONCEPTUAL_RUBRIC_TEMPLATE = {
    "criteria": {
        "technical_accuracy": {
            "weight": 0.4,
            "description": "Correctness of concepts and facts",
            "scoring_guide": {
                "0-2": "Major errors, fundamental misunderstandings",
                "3-5": "Partially correct, missing key concepts",
                "6-8": "Mostly correct, minor gaps",
                "9-10": "Fully correct, comprehensive"
            },
            "key_points": [],  # To be filled from reference answer
            "common_mistakes": []  # To be filled
        },
        "completeness": {
            "weight": 0.3,
            "description": "Coverage of required aspects",
            "scoring_guide": {
                "0-2": "Very incomplete, major aspects missing",
                "3-5": "Some coverage, key points missing",
                "6-8": "Most aspects covered",
                "9-10": "Comprehensive coverage"
            },
            "key_points": []
        },
        "depth": {
            "weight": 0.2,
            "description": "Understanding demonstrated",
            "scoring_guide": {
                "0-2": "Surface-level only",
                "3-5": "Basic understanding",
                "6-8": "Good depth, explains why/how",
                "9-10": "Deep understanding, makes connections"
            }
        },
        "clarity": {
            "weight": 0.1,
            "description": "Communication quality",
            "scoring_guide": {
                "0-2": "Confusing, unclear",
                "3-5": "Somewhat clear",
                "6-8": "Clear and structured",
                "9-10": "Exceptionally clear"
            }
        }
    }
}

CODING_RUBRIC_TEMPLATE = {
    "criteria": {
        "correctness": {
            "weight": 0.5,
            "description": "Does the code work correctly?",
            "scoring_guide": {
                "0-2": "Doesn't work, major bugs",
                "3-5": "Partially works, significant issues",
                "6-8": "Works correctly with minor issues",
                "9-10": "Perfect, handles all cases"
            },
            "key_points": [],  # "Must handle empty array", etc.
            "test_cases": []  # From code_solutions
        },
        "efficiency": {
            "weight": 0.3,
            "description": "Time and space complexity",
            "scoring_guide": {
                "0-2": "Very inefficient (e.g., O(n³) when O(n) possible)",
                "3-5": "Suboptimal but reasonable",
                "6-8": "Good complexity, near optimal",
                "9-10": "Optimal complexity"
            },
            "expected_complexity": {
                "time": "",  # "O(log n)" from code_solutions
                "space": ""  # "O(1)" from code_solutions
            }
        },
        "code_quality": {
            "weight": 0.2,
            "description": "Readability and best practices",
            "scoring_guide": {
                "0-2": "Poor quality, hard to read",
                "3-5": "Acceptable, some issues",
                "6-8": "Good quality, clean code",
                "9-10": "Excellent, professional quality"
            },
            "best_practices": [
                "Meaningful variable names",
                "Proper error handling",
                "Clear logic flow"
            ]
        }
    }
}

SCENARIO_RUBRIC_TEMPLATE = {
    "criteria": {
        "approach": {
            "weight": 0.35,
            "description": "Problem-solving approach",
            "scoring_guide": {
                "0-2": "No clear approach",
                "3-5": "Basic approach, major gaps",
                "6-8": "Solid approach, well-reasoned",
                "9-10": "Excellent, comprehensive approach"
            },
            "key_points": [],  # To be filled from reference answer
            "common_mistakes": []  # To be filled
        },
        "technical_soundness": {
            "weight": 0.35,
            "description": "Technical correctness of solution",
            "scoring_guide": {
                "0-2": "Technically flawed",
                "3-5": "Some technical issues",
                "6-8": "Technically sound",
                "9-10": "Technically excellent"
            },
            "key_points": []  # To be filled from reference answer
        },
        "practicality": {
            "weight": 0.2,
            "description": "Real-world feasibility",
            "scoring_guide": {
                "0-2": "Impractical, unrealistic",
                "3-5": "Somewhat practical",
                "6-8": "Practical and feasible",
                "9-10": "Highly practical, well-considered"
            }
        },
        "communication": {
            "weight": 0.1,
            "description": "Clarity of explanation",
            "scoring_guide": {
                "0-2": "Unclear explanation",
                "3-5": "Somewhat clear",
                "6-8": "Clear explanation",
                "9-10": "Exceptionally clear"
            }
        }
    }
}

MATH_RUBRIC_TEMPLATE = {
    "criteria": {
        "mathematical_correctness": {
            "weight": 0.5,
            "description": "Correctness of derivation/proof",
            "scoring_guide": {
                "0-2": "Major mathematical errors",
                "3-5": "Partially correct, key steps wrong",
                "6-8": "Mostly correct, minor errors",
                "9-10": "Mathematically rigorous and correct"
            },
            "key_points": []  # To be filled from reference answer
        },
        "logical_flow": {
            "weight": 0.3,
            "description": "Step-by-step reasoning",
            "scoring_guide": {
                "0-2": "Illogical or missing steps",
                "3-5": "Some logical gaps",
                "6-8": "Good logical progression",
                "9-10": "Perfect logical flow"
            }
        },
        "notation": {
            "weight": 0.1,
            "description": "Proper mathematical notation",
            "scoring_guide": {
                "0-2": "Incorrect or confusing notation",
                "3-5": "Acceptable notation with issues",
                "6-8": "Correct notation",
                "9-10": "Professional, clear notation"
            }
        },
        "completeness": {
            "weight": 0.1,
            "description": "All steps shown",
            "scoring_guide": {
                "0-2": "Many steps missing",
                "3-5": "Some steps shown",
                "6-8": "Most steps shown",
                "9-10": "All steps clearly shown"
            }
        }
    }
}


class RubricGenerator:
    """
    Automatically generate rubrics from questions and reference answers
    """
    
    def __init__(self, use_local_llm=True):
        if use_local_llm:
            self.llm = ChatOllama(
                model="qwen2.5:7b-instruct-q8_0",
                temperature=0.3,
                base_url="http://localhost:11434"
            )
        else:
            # Placeholder for other LLMs
            self.llm = None
        
        self.templates = {
            'conceptual': CONCEPTUAL_RUBRIC_TEMPLATE,
            'coding': CODING_RUBRIC_TEMPLATE,
            'scenario': SCENARIO_RUBRIC_TEMPLATE,
            'math': MATH_RUBRIC_TEMPLATE
        }
    
    def _clean_json_output(self, text: str) -> str:
        """
        Clean common JSON formatting issues from LLM output
        """
        # Remove markdown code blocks if present
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
        
        # Find the JSON object/array
        # Look for { or [ and find matching closing bracket
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            # Remove trailing characters like ), ], etc. that shouldn't be there
            json_str = re.sub(r'[)\]]+\s*$', '', json_str)
            
            # Fix invalid escape sequences (e.g., LaTeX notation like \beta, \(, etc.)
            # Replace single backslashes with double backslashes, but preserve valid JSON escapes
            # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
            def escape_invalid_backslashes(match):
                char_after = match.group(1)
                # If it's a valid JSON escape, keep it
                if char_after in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
                    return match.group(0)
                # Otherwise, escape the backslash
                return '\\\\' + char_after
            
            json_str = re.sub(r'\\(.)', escape_invalid_backslashes, json_str)
            
            # Add back proper closing if needed
            if json_str.count('{') > json_str.count('}'):
                json_str += '}'
            if json_str.count('[') > json_str.count(']'):
                json_str += ']'
            return json_str.strip()
        
        return text.strip()
    
    def generate_rubric(self, question: dict) -> dict:
        """
        Generate rubric for a single question
        """
        
        # Step 1: Get base template
        q_type = question.get('question_type', 'conceptual')
        rubric = self._get_template(q_type)
        
        # Step 2: Fill in question-specific details
        rubric['question_id'] = question['id']
        
        # Step 3: Extract key points from reference answer
        if question.get('reference_answer'):
            key_points = self._extract_key_points(
                question['text'],
                question['reference_answer'],
                q_type
            )
            rubric = self._populate_key_points(rubric, key_points, q_type)
        
        # Step 4: Extract common mistakes for conceptual and scenario questions only
        if q_type in ['conceptual', 'scenario'] and question.get('reference_answer'):
            common_mistakes = self._extract_common_mistakes(
                question['text'],
                question['reference_answer'],
                q_type
            )
            if common_mistakes:
                rubric = self._populate_common_mistakes(rubric, common_mistakes, q_type)
        
        # Step 5: For coding questions, add test cases
        if q_type == 'coding' and question.get('test_cases'):
            rubric['criteria']['correctness']['test_cases'] = question['test_cases']
        
        # Step 6: For coding questions, add expected complexity
        if q_type == 'coding':
            complexity = self._extract_complexity(question)
            if complexity:
                rubric['criteria']['efficiency']['expected_complexity'] = complexity
        
        return rubric
    
    def _extract_key_points(self, question_text, reference_answer, q_type):
        """
        Use LLM to extract key points from reference answer
        """
        if not self.llm:
            return self._simple_key_point_extraction(reference_answer)

        
        # Define the output structure
        class KeyPoints(BaseModel):
            points: List[str] = Field(description="List of 3-5 key points that must be in a good answer")

        parser = JsonOutputParser(pydantic_object=KeyPoints)

        prompt = PromptTemplate(
            template="""You are an expert technical interviewer creating a grading rubric.
Your task is to identify the crucial technical concepts that a candidate MUST mention to be considered correct.

Question: {question_text}

Reference Answer: {reference_answer}

Identify 3-5 distinct, checkable facts or concepts from the reference answer.
- Each point should be a single, clear sentence.
- Focus on technical accuracy and specific details.
- Avoid generic statements like "Good explanation".

{format_instructions}""",
            input_variables=["question_text", "reference_answer"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "question_text": question_text,
                "reference_answer": reference_answer
            })
            
            # Handle both list output and dict output depending on what the model does
            if isinstance(result, list):
                return result[:5]
            elif isinstance(result, dict) and 'points' in result:
                return result['points'][:5]
            
        except Exception as e:
            # Fallback: Try manual JSON parsing
            try:
                chain_no_parser = prompt | self.llm
                response = chain_no_parser.invoke({
                    "question_text": question_text,
                    "reference_answer": reference_answer
                })
                
                # Extract text content from response
                if hasattr(response, 'content'):
                    response_text = str(response.content)
                else:
                    response_text = str(response)
                
                # Clean and parse JSON
                cleaned_json = self._clean_json_output(response_text)
                result = json.loads(cleaned_json)
                
                # Handle both list output and dict output
                if isinstance(result, list):
                    return result[:5]
                elif isinstance(result, dict) and 'points' in result:
                    return result['points'][:5]
            except Exception as e2:
                print(f"Error extracting key points: {e2}")
        
        # Fallback: Simple extraction
        return self._simple_key_point_extraction(reference_answer)
    
    def _simple_key_point_extraction(self, reference_answer):
        """
        Fallback: Extract key points without LLM
        """
        
        # Split into sentences
        sentences = reference_answer.split('. ')
        
        # Take first 3-4 sentences as key points
        key_points = sentences[:4]
        
        # Clean up
        key_points = [s.strip() + '.' if not s.endswith('.') else s.strip() 
                      for s in key_points if len(s) > 20]
        
        return key_points[:5]
    
    def _extract_complexity(self, question):
        """
        Extract time/space complexity from code solution or reference answer using LLM
        """
        # Check if we have code solution with complexity in metadata (fast path)
        if question.get('time_complexity') and question.get('space_complexity'):
            return {
                'time': question['time_complexity'],
                'space': question['space_complexity']
            }
            
        if not self.llm:
             # Fallback regex extraction if no LLM
            answer = question.get('reference_answer', '')
            import re
            time_match = re.search(r'O\([^)]+\)', answer)
            if time_match:
                return {'time': time_match.group(), 'space': 'O(1)'}
            return None

        # Helper model for extraction
        class Complexity(BaseModel):
            time: str = Field(description="Time complexity, e.g., O(n), O(log n)")
            space: str = Field(description="Space complexity, e.g., O(1), O(n)")

        parser = JsonOutputParser(pydantic_object=Complexity)
        
        prompt = PromptTemplate(
            template="""You are an expert algorithm analyst.
Determine the Time and Space complexity for the solution to this problem.

Question: {question_text}

Reference Solution/Answer: {reference_answer}

Extract the Big O complexities. If not explicitly stated, infer them from the logic.
If you cannot determine them, use "Unknown".

{format_instructions}""",
            input_variables=["question_text", "reference_answer"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "question_text": question.get('text', ''),
                "reference_answer": question.get('reference_answer', '')
            })
            
            return {
                'time': result.get('time', 'Unknown'),
                'space': result.get('space', 'Unknown')
            }
        except Exception:
            # Fallback: Try manual JSON parsing
            try:
                chain_no_parser = prompt | self.llm
                response = chain_no_parser.invoke({
                    "question_text": question.get('text', ''),
                    "reference_answer": question.get('reference_answer', '')
                })
                
                # Extract text content from response
                if hasattr(response, 'content'):
                    response_text = str(response.content)
                else:
                    response_text = str(response)
                
                # Clean and parse JSON
                cleaned_json = self._clean_json_output(response_text)
                result = json.loads(cleaned_json)
                
                return {
                    'time': result.get('time', 'Unknown'),
                    'space': result.get('space', 'Unknown')
                }
            except Exception:
                # Complexity extraction is optional, silently fail
                pass
            
        return None
    
    def _populate_key_points(self, rubric, key_points, q_type):
        """
        Add key points to appropriate criteria based on question type
        """
        
        if q_type == 'conceptual':
            # Add to technical_accuracy and completeness
            if 'technical_accuracy' in rubric['criteria']:
                rubric['criteria']['technical_accuracy']['key_points'] = key_points
            if 'completeness' in rubric['criteria']:
                rubric['criteria']['completeness']['key_points'] = key_points
        
        elif q_type == 'coding':
            # Add to correctness criterion
            if 'correctness' in rubric['criteria']:
                rubric['criteria']['correctness']['key_points'] = key_points
        
        elif q_type == 'scenario':
            # Add to approach and technical_soundness
            if 'approach' in rubric['criteria']:
                rubric['criteria']['approach']['key_points'] = key_points
            if 'technical_soundness' in rubric['criteria']:
                rubric['criteria']['technical_soundness']['key_points'] = key_points
        
        elif q_type == 'math':
            # Add to mathematical_correctness
            if 'mathematical_correctness' in rubric['criteria']:
                rubric['criteria']['mathematical_correctness']['key_points'] = key_points
        
        return rubric
    
    def _extract_common_mistakes(self, question_text, reference_answer, q_type):
        """
        Use LLM to extract common mistakes candidates make for this question
        """
        if not self.llm:
            return []
        
        # Define the output structure
        class CommonMistakes(BaseModel):
            mistakes: List[str] = Field(description="List of 2-4 common mistakes candidates make")

        parser = JsonOutputParser(pydantic_object=CommonMistakes)

        prompt = PromptTemplate(
            template="""You are an expert technical interviewer.
Based on this question and its reference answer, identify common mistakes or misconceptions candidates often have.

Question Type: {question_type}
Question: {question_text}

Reference Answer: {reference_answer}

Identify 2-4 common mistakes, misconceptions, or pitfalls candidates might fall into for this {question_type} question.
- Each mistake should be specific and actionable for grading.
- For coding: focus on implementation errors, edge cases, algorithmic mistakes
- For conceptual: focus on misunderstandings, incorrect definitions, confused concepts
- For scenario: focus on missing considerations, impractical solutions, overlooked constraints
- For math: focus on calculation errors, wrong formulas, logical gaps

{format_instructions}""",
            input_variables=["question_text", "reference_answer", "question_type"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "question_text": question_text,
                "reference_answer": reference_answer,
                "question_type": q_type
            })
            
            # Handle both list output and dict output
            if isinstance(result, list):
                return result[:4]
            elif isinstance(result, dict) and 'mistakes' in result:
                return result['mistakes'][:4]
            
        except Exception:
            # Fallback: Try manual JSON parsing
            try:
                chain_no_parser = prompt | self.llm
                response = chain_no_parser.invoke({
                    "question_text": question_text,
                    "reference_answer": reference_answer,
                    "question_type": q_type
                })
                
                # Extract text content from response
                if hasattr(response, 'content'):
                    response_text = str(response.content)
                else:
                    response_text = str(response)
                
                # Clean and parse JSON
                cleaned_json = self._clean_json_output(response_text)
                result = json.loads(cleaned_json)
                
                # Handle both list output and dict output
                if isinstance(result, list):
                    return result[:4]
                elif isinstance(result, dict) and 'mistakes' in result:
                    return result['mistakes'][:4]
            except Exception:
                # Common mistakes are optional, silently fail
                pass
        
        return []
    
    def _populate_common_mistakes(self, rubric, common_mistakes, q_type):
        """
        Add common mistakes to appropriate criteria based on question type
        Only for conceptual and scenario questions
        """
        
        if q_type == 'conceptual':
            # Add to technical_accuracy
            if 'technical_accuracy' in rubric['criteria']:
                rubric['criteria']['technical_accuracy']['common_mistakes'] = common_mistakes
        
        elif q_type == 'scenario':
            # Add to approach
            if 'approach' in rubric['criteria']:
                rubric['criteria']['approach']['common_mistakes'] = common_mistakes
        
        return rubric


    
    def _get_template(self, q_type):
        """
        Get and deep copy the appropriate template
        """
        template = self.templates.get(q_type, self.templates['conceptual'])
        return copy.deepcopy(template)

    def generate_all_rubrics(self, questions):
        """
        Generate rubrics for all questions
        """
        
        rubrics = {}
        
        print(f"Generating rubrics for {len(questions)} questions...")
        
        for i, question in enumerate(questions):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(questions)}")
            
            rubric = self.generate_rubric(question)
            rubrics[question['id']] = rubric
        
        print(f"✓ Generated {len(rubrics)} rubrics")
        
        return rubrics
    
def identify_questions_needing_manual_review(questions):
    """
    Flag questions that need manual rubric customization
    """
    
    needs_review = []
    
    for question in questions:
        # Flag 1: Hard questions
        if question['difficulty'] == 'hard':
            needs_review.append({
                'id': question['id'],
                'reason': 'Hard question - needs detailed rubric'
            })
        
        # Flag 2: No reference answer
        elif not question.get('reference_answer'):
            needs_review.append({
                'id': question['id'],
                'reason': 'No reference answer'
            })
        
        # Flag 3: Math/derivation questions
        elif question['question_type'] == 'math':
            needs_review.append({
                'id': question['id'],
                'reason': 'Math question - verify key steps'
            })
        
        # Flag 4: Complex scenario questions
        elif (question['question_type'] == 'scenario' and 
              'design' in question['text'].lower()):
            needs_review.append({
                'id': question['id'],
                'reason': 'System design - needs custom criteria'
            })
    
    return needs_review

def validate_rubric(rubric):
    """
    Check if rubric is well-formed
    """
    
    issues = []
    
    # Check 1: Has criteria
    if not rubric.get('criteria'):
        issues.append("Missing criteria")
        return {'valid': False, 'issues': issues}
    
    # Check 2: Weights sum to 1.0
    total_weight = sum(
        criteria.get('weight', 0) 
        for criteria in rubric['criteria'].values()
    )
    
    if abs(total_weight - 1.0) > 0.01:
        issues.append(f"Weights sum to {total_weight}, not 1.0")
    
    # Check 3: Each criterion has scoring guide
    for name, criteria in rubric['criteria'].items():
        if not criteria.get('scoring_guide'):
            issues.append(f"Criterion '{name}' missing scoring_guide")
        
        # Check scoring guide has all ranges
        scoring_guide = criteria.get('scoring_guide', {})
        required_ranges = ['0-2', '3-5', '6-8', '9-10']
        for range_key in required_ranges:
            if range_key not in scoring_guide:
                issues.append(f"Criterion '{name}' missing range '{range_key}'")
    
    # Check 4: Has at least some key points
    has_key_points = any(
        len(criteria.get('key_points', [])) > 0
        for criteria in rubric['criteria'].values()
    )
    
    if not has_key_points:
        issues.append("No key points defined in any criterion")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }


def validate_all_rubrics(rubrics):
    """
    Validate all generated rubrics
    """
    
    print("Validating rubrics...")
    
    invalid = []
    
    for question_id, rubric in rubrics.items():
        validation = validate_rubric(rubric)
        
        if not validation['valid']:
            invalid.append({
                'question_id': question_id,
                'issues': validation['issues']
            })
    
    if invalid:
        print(f"⚠️  {len(invalid)} rubrics have issues")
        for item in invalid[:5]:  # Show first 5
            print(f"  {item['question_id']}: {', '.join(item['issues'])}")
    else:
        print("✓ All rubrics valid")
    
    return {
        'total': len(rubrics),
        'valid': len(rubrics) - len(invalid),
        'invalid': len(invalid),
        'invalid_details': invalid
    }


def main():
    base_dir = Path(__file__).parent.parent

    datasets_path = base_dir / "data/datasets/processed/interview_questions"

    files = [
        "filtered_github_kaggle_iqs.json",
        "interview_questions_chip.json",
        "interview_questions.json",
        "leetcode_questions.json",
        "llm_generated_iqs.json",
        "system_design_iqs.json"
    ]
    questions = []
    for file in files:
        print(f"  - Loading {file}...")
        with open(datasets_path / file, "r") as fp:
            data = json.load(fp)
        questions.extend(data)

    
    print(f"Loaded {len(questions)} questions")
    
    # Initialize generator
    generator = RubricGenerator(use_local_llm=True)
    
    # Generate rubrics
    rubrics = generator.generate_all_rubrics(questions)
    
    # Validate
    validation_results = validate_all_rubrics(rubrics)
    
    # Save
    output_file = 'data/rubrics/all_rubrics.json'
    with open(output_file, 'w') as f:
        json.dump(rubrics, f, indent=2)
    
    print(f"\n✓ Saved {len(rubrics)} rubrics to {output_file}")
    
    # Identify questions needing manual review
    needs_review = identify_questions_needing_manual_review(questions)
    
    if needs_review:
        print(f"\n⚠️  {len(needs_review)} questions recommended for manual review")
        print("\nSaving review list...")
        
        with open('data/rubrics/manual_review_needed.json', 'w') as f:
            json.dump(needs_review, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RUBRIC GENERATION COMPLETE")
    print("="*60)
    print(f"Total rubrics:     {validation_results['total']}")
    print(f"Valid:             {validation_results['valid']}")
    print(f"Needs fixing:      {validation_results['invalid']}")
    print(f"Manual review:     {len(needs_review)}")

if __name__ == "__main__":
    main()