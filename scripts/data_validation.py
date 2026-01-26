import json
import random
import os
from pathlib import Path
from warnings import warn
from numpy import mean, std
from sentence_transformers import SentenceTransformer, util
from detoxify import Detoxify
import re
from typing import Dict, List
from collections import Counter, defaultdict
from scipy.stats import chisquare


topics = [
    "linear algebra", "vectors", "tensors", "eigen values", "matrix decomposition", "bayes", "null hypothesis", "variance", "standard deviation",
    "p-value", "chi square test", "anova", "correlation", "outlier detection", "data normalization", "data standardization", "histograms", "box plot",
    "heat maps", "scatter plot", "feature selection", "pca", "lda", "autoencoders", "supervised learning", "unsupervised learning", "self supervised learning",
    "reinforcement learning", "cross validation", "bias variance", "underfitting", "overfitting", "regularization", "regression", "classification", "logistic regression",
    "svm", "knn", "decision tree", "random forest", "boosting", "xgboost", "clustering", "k-means", "dbscan", "confusion matrix", "accuracy", "metrics", "precision", "recall",
    "auc", "mse", "rmse", "gradient descent", "neural networks", "activation functions", "loss function", "adam", "rmsprop", "convolution", "recurrent neural network",
    "lstm", "transformers", "attention", "text preprocessing", "nlp", "deep learning", "machine learning", "algorithm", "model", "optimization"
]

PII_PATTERNS = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "phone": re.compile(
        r"\b(\+?\d{1,3}[\s-]?)?\d{10}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d[ -]*?){13,16}\b"
    ),
    "aadhaar": re.compile(
        r"\b\d{4}\s?\d{4}\s?\d{4}\b"
    ),
    "ssn": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "ip_address": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
    "dob": re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    ),
}

print("Loading models...")
print("  - Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("  ✓ SentenceTransformer loaded")
print("  - Loading Detoxify model (original)...")
toxicity_model = Detoxify('original', device='cuda')
print("  ✓ Detoxify loaded")
print("Models loaded successfully!\n")

ml_interview_embeddings = model.encode("machine learning interview question")

def compute_semantic_similarity(text, reference=None):
    text_embeddings = model.encode(text)
    if reference:
        reference_embeddings = model.encode(reference)
    else:
        reference_embeddings = ml_interview_embeddings
    return util.cos_sim(text_embeddings, reference_embeddings)


def contains_pii(text: str) -> bool:
    if not text:
        return False

    matches = {}

    for pii_type, pattern in PII_PATTERNS.items():
        found = pattern.findall(text)
        if found:
            matches[pii_type] = found

    return bool(matches)

class DataQuality:
    def __init__(self, input_data):
        self.input_data = input_data

    def check_relevance(self):
        print("  → Checking relevance...")
        irrelevant_count = 0
        irrelevant_data = []
        for question in self.input_data:
            text = question.get('text', '').lower()
            
            keyword_matches = sum(1 for kw in topics if kw in text)

            relevance_score = compute_semantic_similarity(text)

            if keyword_matches == 0 and relevance_score < 0.5:
                irrelevant_count += 1
                irrelevant_data.append(question)

        relevance_rate = 1 - (irrelevant_count / len(self.input_data))

        return {
            "status": "PASS" if relevance_rate > 0.80 else "FAIL",
            "irrelevant_data": irrelevant_data

        } 
        
    def check_consistency(self):
        print("  → Checking consistency...")
        REQUIRED_FIELDS = ['id', 'text', 'difficulty', 'topic', 'question_type']

        VALID_DIFFICULTIES = ['easy', 'medium', 'hard']
        VALID_TYPES = ['conceptual', 'coding', 'scenario', 'math']

        issues = []
        for i, question in enumerate(self.input_data):
            q_id = question['id']
            text = question.get('text', '')
            
            # check 1: all fields
            for field in REQUIRED_FIELDS:
                if field not in question:
                    issues.append({
                        'id': q_id,
                        'reason': f"Missing field '{field}'",
                        'text': text[:50]
                    })

            # check 2: valid enum values
            if question.get('difficulty') not in VALID_DIFFICULTIES:
                issues.append({
                    'id': q_id,
                    'reason': f"Invalid difficulty '{question.get('difficulty')}'",
                    'text': text[:50]
                })
            if question.get('question_type') not in VALID_TYPES:
                issues.append({
                    'id': q_id,
                    'reason': f"Invalid question type '{question.get('question_type')}'",
                    'text': text[:50]
                })

            # Check 3: Data types correct
            if not isinstance(question.get('text'), str):
                issues.append({
                    'id': q_id,
                    'reason': "'text' must be string",
                    'text': str(question.get('text'))[:50]
                })
        
            if not isinstance(question.get('tags'), list):
                issues.append({
                    'id': q_id,
                    'reason': "'tags' must be list",
                    'text': text[:50]
                })

            # Check 4: Text quality
            if len(text) < 10:
                issues.append({
                    'id': q_id,
                    'reason': "Text too short",
                    'text': text[:50]
                })
        
            if len(text) > 1000:
                issues.append({
                    'id': q_id,
                    'reason': "Text too long (>1000 chars)",
                    'text': text[:50]
                })
        consistency_rate = 1 - (len(issues) / len(self.input_data))

        return {
            'pass': consistency_rate >=0.95,
            'score': consistency_rate,
            'issue': issues
        }
        
    def check_correctness(self):
        print("  → Checking correctness...")
        suspicious = []

        for question in self.input_data:
            text = question.get('text', '')

            # check 1: gibberish detection (vowel ratio)
            vowels = sum(1 for c in text.lower() if c in 'aeiou')
            vowel_ratio = vowels / len(text) if text else 0

            if vowel_ratio < 0.2:
                suspicious.append({
                    'id': question['id'],
                    'reason': 'Too few vowels',
                    'text': text[:50]
                })
                continue

            # Check 2: Excessive special characters
            special_chars = sum(1 for c in text if not c.isalnum() and c not in ' .,?!-')
            special_ratio = special_chars / len(text)
            
            if special_ratio > 0.3:
                suspicious.append({
                    'id': question['id'],
                    'reason': 'Too many special characters',
                    'text': text[:50]
                })
                continue

            # Check 3: Repeated words (indicates scraping error)
            words = text.lower().split()
            word_set = set(words)
            if len(words) > 10 and len(word_set) / len(words) < 0.3:
                suspicious.append({
                    'id': question['id'],
                    'reason': 'Too many repeated words',
                    'text': text[:50]
                })
                continue
        correctness_rate = 1 - (len(suspicious)/ len(self.input_data))

        return {
            "pass": correctness_rate >= 0.90,
            "score": correctness_rate,
            "suspicious_items": suspicious
        }


    def check_uniqueness(self):
        print("  → Checking uniqueness (generating embeddings)...")
        texts = [q.get('text', '') for q in self.input_data]
        embeddings = model.encode(texts)
        print("    Embeddings generated, comparing for duplicates...")

        similarity_matrix = util.cos_sim(embeddings, embeddings)
        DUPLICATE_THRESHOLD = 0.85

        duplicates = []
        keep_indices = set(range(len(self.input_data)))

        for i in range(len(self.input_data)):
            if i not in keep_indices:
                continue

            for j in range(i+1, len(self.input_data)):
                if j not in keep_indices:
                    continue
                if similarity_matrix[i][j] > DUPLICATE_THRESHOLD:
                    # Mark as duplicate
                    duplicates.append({
                        'q1_id': self.input_data[i]['id'],
                        'q2_id': self.input_data[j]['id'],
                        'similarity': similarity_matrix[i][j],
                        'q1_text': self.input_data[i].get('text', '')[:50],
                        'q2_text': self.input_data[j].get('text', '')[:50]
                    })

                    # Keep the longer/better one
                    text1 = self.input_data[i].get('text', '')
                    text2 = self.input_data[j].get('text', '')
                    if len(text1) >= len(text2):
                        keep_indices.discard(j)
                    else:
                        keep_indices.discard(i)
                        break
        
        unique_questions = [self.input_data[i] for i in sorted(keep_indices)]
        
        # Identify removed questions
        removed_indices = set(range(len(self.input_data))) - keep_indices
        to_remove = []
        for idx in removed_indices:
            to_remove.append({
                'id': self.input_data[idx]['id'],
                'reason': 'Duplicate question',
                'text': self.input_data[idx].get('text', '')[:50]
            })
        uniqueness_rate = len(unique_questions) / len(self.input_data)

        return {
        'original_count': len(self.input_data),
        'unique_count': len(unique_questions),
        'duplicates_found': len(duplicates),
        'uniqueness_rate': uniqueness_rate,
        'uniqueness_rate': uniqueness_rate,
        'duplicate_pairs': duplicates,
        'to_remove': to_remove
    }

    def check_compliance(self):
        print("  → Checking compliance (toxicity & PII)...")
        violations = []
        for question in self.input_data:
            ref_answer = question.get('reference_answer', '')
            if not isinstance(ref_answer, str):
                ref_answer = str(ref_answer)
            
            text_val = question.get('text', '')
            if not isinstance(text_val, str):
                text_val = str(text_val)

            text = text_val + ' ' + ref_answer
            # check 1: toxicity check
            toxicity_results = toxicity_model.predict(text)
            toxicity_score = round(float(toxicity_results['toxicity']), 2)
            if toxicity_score > 0.5:
                violations.append({
                    'id': question['id'],
                    'type': 'toxicity',
                    'score': toxicity_score
                })

            # check 2: PII check
            if contains_pii(text):
                violations.append({
                    'id': question['id'],
                    'type': 'pii_leak'
                })

        compliance_rate = 1 - (len(violations)/len(self.input_data))

        return {
            'pass': compliance_rate >= 0.98,
            'score': compliance_rate,
            'violations': violations
        }
class DataCoverage:
    def __init__(self, input_data):
        self.input_data = input_data

    def check_topic_diversity(self):
        print("  → Checking topic diversity...")
        TARGET_DISTRIBUTION = {
        'ml_fundamentals': 0.25,
        'deep_learning': 0.20,
        'classical_algorithms': 0.15,
        'optimization': 0.10,
        'evaluation': 0.10,
        'system_design': 0.10,
        'statistics': 0.10
    }
        # Count actual distribution
        topic_counts = Counter(q.get('topic', 'unknown') for q in self.input_data)
        total = len(self.input_data)
        
        actual_distribution = {
            topic: count/total 
            for topic, count in topic_counts.items()
        }
        
        # Compare with target
        coverage_scores = {}
        for topic, target_pct in TARGET_DISTRIBUTION.items():
            actual_pct = actual_distribution.get(topic, 0)
            
            # How close to target?
            deviation = abs(target_pct - actual_pct)
            
            coverage_scores[topic] = {
                'target': target_pct,
                'actual': actual_pct,
                'deviation': deviation,
                'count': topic_counts.get(topic, 0),
                'status': 'OK' if deviation < 0.05 else 'NEEDS_MORE'
            }
        
        # Chi-square test for uniform distribution
        # Only test topics that are in our target distribution
        observed = []
        expected = []
        
        for topic, target_pct in TARGET_DISTRIBUTION.items():
            observed.append(topic_counts.get(topic, 0))
            expected.append(target_pct)
            
        total_observed = sum(observed)
        if total_observed > 0:
            # Normalize expected to match sum of observed
            total_expected_pct = sum(expected)
            expected = [e/total_expected_pct * total_observed for e in expected]
            chi_square_stat, p_value = chisquare(observed, f_exp=expected)
        else:
            chi_square_stat, p_value = 0, 0 # Could not compute
        
        return {
            'pass': p_value > 0.05,  # Not significantly skewed
            'p_value': p_value,
            'coverage_by_topic': coverage_scores
        }
    
    def check_difficulty_distribution(self):
        print("  → Checking difficulty distribution...")
        TARGET = {
            'easy': 0.30,    # 30%
            'medium': 0.50,  # 50%
            'hard': 0.20     # 20%
        }
        
        difficulty_counts = Counter(q.get('difficulty', 'unknown') for q in self.input_data)
        total = len(self.input_data)
        
        actual = {
            diff: count/total 
            for diff, count in difficulty_counts.items()
        }
        
        # Validate distribution
        for difficulty, target_pct in TARGET.items():
            actual_pct = actual.get(difficulty, 0)
            deviation = abs(target_pct - actual_pct)
            
            if deviation > 0.15:  # More than 15% deviation
                return {
                    'pass': False,
                    'issue': f"{difficulty} questions: {actual_pct:.1%} (target: {target_pct:.1%})"
                }
        
        return {
            'pass': True,
            'distribution': actual
        }
    
    def check_type_distribution(self):
        print("  → Checking type distribution...")
        TARGET = {
            'conceptual': 0.40,  # 40%
            'scenario': 0.30, # 30%
            "coding": 0.20, # 20%
            'math': 0.10 # 10%
        }
        
        type_counts = Counter(q.get('question_type', 'unknown') for q in self.input_data)
        total = len(self.input_data)
        
        actual = {
            diff: count/total 
            for diff, count in type_counts.items()
        }
        
        # Validate distribution
        for qtype, target_pct in TARGET.items():
            actual_pct = actual.get(qtype, 0)
            deviation = abs(target_pct - actual_pct)
            
            if deviation > 0.15:  # More than 15% deviation
                return {
                    'pass': False,
                    'issue': f"{qtype} questions: {actual_pct:.1%} (target: {target_pct:.1%})"
                }
        
        # Additional: Check if each topic has type variety
        for topic in set(q.get('topic', 'unknown') for q in self.input_data):
            topic_questions = [q for q in self.input_data if q.get('topic') == topic]
            types_in_topic = set(q.get('question_type') for q in topic_questions)
            
            if len(types_in_topic) < 2:
                warn(f"Topic '{topic}' has only one question type: {types_in_topic}")

        return {
            'pass': True,
            'distribution': actual
        }
    def check_embedding_diversity(self):
        print("  → Checking embedding diversity (generating embeddings)...")
        # Generate embeddings
        texts = [q.get('text', '') for q in self.input_data]
        embeddings = model.encode(texts)
        print("    Computing pairwise distances...")
        
        # Compute pairwise cosine distances
        distances = []
        n = len(embeddings)
        
        for i in range(n):
            for j in range(i+1, n):
                distance = 1 - util.cos_sim([embeddings[i]], [embeddings[j]])[0][0]
                distances.append(distance)
        
        avg_distance = mean(distances)
        std_distance = std(distances)
        
        # Interpretation:
        # High avg distance = diverse questions
        # Low avg distance = questions too similar
        
        DIVERSITY_THRESHOLD = 0.6  # Minimum acceptable average distance
        
        return {
            'pass': avg_distance >= DIVERSITY_THRESHOLD,
            'avg_distance': avg_distance,
            'std_distance': std_distance,
            'interpretation': 'Diverse' if avg_distance >= 0.7 else 
                            'Moderate' if avg_distance >= 0.6 else 
                            'Too similar'
        }
    def check_topic_gaps(self):
        print("  → Checking topic gaps...")
        # Comprehensive ML/DS taxonomy
        ML_TAXONOMY = {
            'ml_fundamentals': [
                'supervised_learning', 'unsupervised_learning', 'reinforcement_learning',
                'overfitting', 'underfitting', 'bias_variance_tradeoff', 
                'cross_validation', 'regularization'
            ],
            'optimization': [
                'gradient_descent', 'sgd', 'momentum', 'adam', 'learning_rate',
                'batch_normalization', 'gradient_clipping'
            ],
            'deep_learning': [
                'neural_networks', 'cnn', 'rnn', 'lstm', 'transformer',
                'attention', 'backpropagation', 'activation_functions'
            ],
            # ... more categories
        }
        
        MIN_QUESTIONS_PER_SUBTOPIC = 3
        
        gaps = []
        
        for category, subtopics in ML_TAXONOMY.items():
            for subtopic in subtopics:
                # Count questions covering this subtopic
                count = sum(
                    1 for q in self.input_data 
                    if subtopic in q.get('tags', []) or 
                    subtopic in q.get('text', '').lower().replace(' ', '_')
                )
                
                if count < MIN_QUESTIONS_PER_SUBTOPIC:
                    gaps.append({
                        'category': category,
                        'subtopic': subtopic,
                        'current_count': count,
                        'needed': MIN_QUESTIONS_PER_SUBTOPIC - count
                    })
        
        # Prioritize gaps by importance
        IMPORTANT_TOPICS = [
            'gradient_descent', 'overfitting', 'neural_networks', 
            'regularization', 'cross_validation', 'classification', 'supervised learning',
            'statistics', 'evaluation metric'
        ]
        
        critical_gaps = [
            gap for gap in gaps 
            if gap['subtopic'] in IMPORTANT_TOPICS
        ]
        
        return {
            'total_gaps': len(gaps),
            'critical_gaps': len(critical_gaps),
            'gap_details': sorted(gaps, key=lambda x: x['needed'], reverse=True),
            'pass': len(critical_gaps) == 0
        }


class DataQuantity:
    def __init__(self, input_data):
        self.input_data = input_data
    
    def check_overall_quantity(self):
        print("  → Checking overall quantity...")
        REQUIREMENTS = {
            'minimum': 300,
            'recommended': 500,
            'ideal': 1000
        }
        
        actual_count = len(self.input_data)
        
        if actual_count < REQUIREMENTS['minimum']:
            status = 'INSUFFICIENT'
            level = 'minimum'
        elif actual_count < REQUIREMENTS['recommended']:
            status = 'MVP_READY'
            level = 'minimum'
        elif actual_count < REQUIREMENTS['ideal']:
            status = 'GOOD'
            level = 'recommended'
        else:
            status = 'EXCELLENT'
            level = 'ideal'
        
        return {
            'pass': actual_count >= REQUIREMENTS['minimum'],
            'count': actual_count,
            'status': status,
            'level': level,
            'progress': f"{actual_count}/{REQUIREMENTS['ideal']}"
        }
    def check_quantity_by_category(self):
        print("  → Checking quantity by category...")
        MIN_PER_TOPIC = 30
        MIN_PER_DIFFICULTY = 50
        MIN_PER_TYPE = 50
        
        # Group by topic
        by_topic = defaultdict(list)
        for q in self.input_data:
            by_topic[q.get('topic', 'unknown')].append(q)
        
        # Group by difficulty
        by_difficulty = defaultdict(list)
        for q in self.input_data:
            by_difficulty[q.get('difficulty', 'unknown')].append(q)
        
        # Group by type
        by_type = defaultdict(list)
        for q in self.input_data:
            by_type[q.get('question_type', 'unknown')].append(q)
        
        insufficient = []
        
        # Check topics
        for topic, qs in by_topic.items():
            if len(qs) < MIN_PER_TOPIC:
                insufficient.append({
                    'category': 'topic',
                    'name': topic,
                    'current': len(qs),
                    'required': MIN_PER_TOPIC,
                    'deficit': MIN_PER_TOPIC - len(qs)
                })
        
        # Check difficulties
        for difficulty, qs in by_difficulty.items():
            if len(qs) < MIN_PER_DIFFICULTY:
                insufficient.append({
                    'category': 'difficulty',
                    'name': difficulty,
                    'current': len(qs),
                    'required': MIN_PER_DIFFICULTY,
                    'deficit': MIN_PER_DIFFICULTY - len(qs)
                })
        
        # Check types
        for q_type, qs in by_type.items():
            if len(qs) < MIN_PER_TYPE:
                insufficient.append({
                    'category': 'type',
                    'name': q_type,
                    'current': len(qs),
                    'required': MIN_PER_TYPE,
                    'deficit': MIN_PER_TYPE - len(qs)
                })
        
        return {
            'pass': len(insufficient) == 0,
            'insufficient_categories': insufficient,
            'total_deficit': sum(item['deficit'] for item in insufficient)
        }
    def check_interview_capacity(self):
        print("  → Checking interview capacity (running simulations)...")
        QUESTIONS_PER_INTERVIEW = 10
        MIN_UNIQUE_INTERVIEWS = 20  # Want at least 20 different interviews
        
        simulations = []
        
        for interview_num in range(MIN_UNIQUE_INTERVIEWS):
            # Simulate interview question selection
            selected = []
            available = self.input_data.copy()
            topics_covered = set()
            
            for turn in range(QUESTIONS_PER_INTERVIEW):
                
                # Filter: Avoid repeating topics too soon
                if len(topics_covered) > 0:
                    candidates = [
                        q for q in available 
                        if q.get('topic') not in topics_covered
                    ]
                else:
                    candidates = available
                
                if not candidates:
                    # Can't find unique question
                    simulations.append({
                        'interview_num': interview_num,
                        'completed': False,
                        'questions_asked': len(selected),
                        'reason': 'Ran out of unique questions'
                    })
                    break
                
                # Select question
                question = random.choice(candidates)
                selected.append(question)
                topics_covered.add(question.get('topic', 'unknown'))
                
                # Remove from available
                available.remove(question)
            
            if len(selected) == QUESTIONS_PER_INTERVIEW:
                simulations.append({
                    'interview_num': interview_num,
                    'completed': True,
                    'questions_asked': QUESTIONS_PER_INTERVIEW
                })
        
        successful = sum(1 for s in simulations if s['completed'])
        
        return {
            'pass': successful >= MIN_UNIQUE_INTERVIEWS,
            'successful_interviews': successful,
            'required': MIN_UNIQUE_INTERVIEWS,
            'total_simulated': len(simulations),
            'details': simulations
        }

class ValidateData:
    def __init__(self):
        self.final_data = []
    @staticmethod
    def load_json(file):
        with open(file, "r") as fp:
            data = json.load(fp)

        return data

    def combine_data(self, datasets_path, files):
        print("\nLoading datasets...")
        for file in files:
            print(f"  - Loading {file}...")
            data = ValidateData.load_json(datasets_path / file)
            self.final_data.extend(data)
            print(f"    Loaded {len(data)} questions")
        print(f"\nTotal questions loaded: {len(self.final_data)}\n")

    def validate(self):
        data_quality = DataQuality(self.final_data)
        data_coverage = DataCoverage(self.final_data)
        data_quantity = DataQuantity(self.final_data)
        results = {
            'quality': {},
            'coverage': {},
            'quantity': {}
        }
        
        print("="*60)
        print("DATA VALIDATION REPORT")
        print("="*60)
        
        # ===== QUALITY CHECKS =====
        print("\n1. QUALITY CHECKS")
        print("-"*60)
        
        # results['quality']['relevance'] = data_quality.check_relevance()
        results['quality']['consistency'] = data_quality.check_consistency()
        results['quality']['correctness'] = data_quality.check_correctness()
        results['quality']['uniqueness'] = data_quality.check_uniqueness()
        results['quality']['compliance'] = data_quality.check_compliance()
        
        print("  ✓ Quality checks completed")
        quality_score = self._calculate_quality_score(results['quality'])
        
        # ===== COVERAGE CHECKS =====
        print("\n2. COVERAGE CHECKS")
        print("-"*60)
        
        results['coverage']['topic_distribution'] = data_coverage.check_topic_diversity()
        results['coverage']['difficulty_distribution'] = data_coverage.check_difficulty_distribution()
        results['coverage']['type_distribution'] = data_coverage.check_type_distribution()
        results['coverage']['embedding_diversity'] = data_coverage.check_embedding_diversity()
        results['coverage']['gap_analysis'] = data_coverage.check_topic_gaps()
        
        print("  ✓ Coverage checks completed")
        coverage_score = self._calculate_coverage_score(results['coverage'])
        
        # ===== QUANTITY CHECKS =====
        print("\n3. QUANTITY CHECKS")
        print("-"*60)
        
        results['quantity']['overall'] = data_quantity.check_overall_quantity()
        results['quantity']['by_category'] = data_quantity.check_quantity_by_category()
        results['quantity']['interview_capacity'] = data_quantity.check_interview_capacity()
        
        print("  ✓ Quantity checks completed")
        
        quantity_score = self._calculate_quantity_score(results['quantity'])
        
        # ===== OVERALL ASSESSMENT =====
        print("\n" + "="*60)
        print("OVERALL ASSESSMENT")
        print("="*60)
        
        overall_score = (quality_score + coverage_score + quantity_score) / 3
        
        print(f"Quality Score:   {quality_score:.1%}")
        print(f"Coverage Score:  {coverage_score:.1%}")
        print(f"Quantity Score:  {quantity_score:.1%}")
        print(f"\nOverall Score:   {overall_score:.1%}")
        
        if overall_score >= 0.85:
            status = "✓ READY FOR MVP"
        elif overall_score >= 0.70:
            status = "⚠️ NEEDS MINOR IMPROVEMENTS"
        else:
            status = "✗ NEEDS SIGNIFICANT WORK"
        
        print(f"\nStatus: {status}")
        
        # ===== RECOMMENDATIONS =====
        recommendations = self._generate_recommendations(results)
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        for rec in recommendations:
            print(f"• {rec}")
        
        return {
            'results': results,
            'scores': {
                'quality': quality_score,
                'coverage': coverage_score,
                'quantity': quantity_score,
                'overall': overall_score
            },
            'status': status,
            'recommendations': recommendations
        }
    def _generate_recommendations(self, results):
        recommendations = []
        
        # Quality recommendations
        if results['quality']['uniqueness']['uniqueness_rate'] < 0.85:
            recommendations.append(
                f"Remove {results['quality']['uniqueness']['duplicates_found']} duplicates"
            )
        
        # Coverage recommendations
        gaps = results['coverage']['gap_analysis']['gap_details']
        if gaps:
            top_gaps = sorted(gaps, key=lambda x: x['needed'], reverse=True)[:5]
            for gap in top_gaps:
                recommendations.append(
                    f"Add {gap['needed']} questions on '{gap['subtopic']}'"
                )
        
        # Quantity recommendations
        if results['quantity']['overall']['count'] < 500:
            needed = 500 - results['quantity']['overall']['count']
            recommendations.append(f"Collect {needed} more questions to reach recommended threshold")
        
        return recommendations
    
    def _calculate_quality_score(self, quality):
        scores = []
        
        # Relevance
        # if quality['relevance']['status'] == 'PASS':
        #     scores.append(1.0)
        # else:
        #     scores.append(0.5)
        
        # Consistency
        scores.append(quality['consistency']['score'])
        
        # Correctness
        scores.append(quality['correctness']['score'])
        
        # Uniqueness
        scores.append(quality['uniqueness']['uniqueness_rate'])
        
        # Compliance
        scores.append(quality['compliance']['score'])
        
        return sum(scores) / len(scores)
    
    def _calculate_coverage_score(self, coverage):
        scores = []
        
        # Topic diversity
        scores.append(1.0 if coverage['topic_distribution']['pass'] else 0.7)
        
        # Difficulty distribution
        scores.append(1.0 if coverage['difficulty_distribution']['pass'] else 0.7)
        
        # Type distribution
        scores.append(1.0 if coverage['type_distribution']['pass'] else 0.7)
        
        # Embedding diversity
        scores.append(1.0 if coverage['embedding_diversity']['pass'] else 0.6)
        
        # Gap analysis
        scores.append(1.0 if coverage['gap_analysis']['pass'] else 0.5)
        
        return sum(scores) / len(scores)
    
    def _calculate_quantity_score(self, quantity):
        scores = []
        
        # Overall quantity
        overall = quantity['overall']
        if overall['status'] == 'EXCELLENT':
            scores.append(1.0)
        elif overall['status'] == 'GOOD':
            scores.append(0.85)
        elif overall['status'] == 'MVP_READY':
            scores.append(0.7)
        else:
            scores.append(0.5)
        
        # By category
        scores.append(1.0 if quantity['by_category']['pass'] else 0.6)
        
        # Interview capacity
        capacity = quantity['interview_capacity']
        capacity_ratio = capacity['successful_interviews'] / capacity['required']
        scores.append(min(1.0, capacity_ratio))
        
        return sum(scores) / len(scores)
        
        

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

    validate = ValidateData()
    validate.combine_data(datasets_path, files)

    validation_results = validate.validate()

    # Create output directory for validation artifacts
    output_dir = base_dir / "data/validation_recheck_output"
    output_dir.mkdir(exist_ok=True)
    
    # Write main validation report
    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\nValidation report saved to: {report_path}")
    
    # Extract and save detailed data artifacts
    results = validation_results['results']
    
    # Quality artifacts
    quality_dir = output_dir / "quality"
    quality_dir.mkdir(exist_ok=True)
    
    # if results['quality']['relevance'].get('irrelevant_data'):
    #     with open(quality_dir / "irrelevant_data.json", 'w') as f:
    #         json.dump(results['quality']['relevance']['irrelevant_data'], f, indent=2)
    #     print(f"  - Irrelevant data saved to: {quality_dir / 'irrelevant_data.json'}")
    
    if results['quality']['consistency'].get('issue'):
        with open(quality_dir / "consistency_issues.json", 'w') as f:
            json.dump(results['quality']['consistency']['issue'], f, indent=2)
        print(f"  - Consistency issues saved to: {quality_dir / 'consistency_issues.json'}")
    
    if results['quality']['correctness'].get('suspicious_items'):
        with open(quality_dir / "suspicious_items.json", 'w') as f:
            json.dump(results['quality']['correctness']['suspicious_items'], f, indent=2)
        print(f"  - Suspicious items saved to: {quality_dir / 'suspicious_items.json'}")
    
    if results['quality']['uniqueness'].get('duplicate_pairs'):
        with open(quality_dir / "duplicate_pairs.json", 'w') as f:
            json.dump(results['quality']['uniqueness']['duplicate_pairs'], f, indent=2, default=str)
        print(f"  - Duplicate pairs saved to: {quality_dir / 'duplicate_pairs.json'}")
    
    if results['quality']['compliance'].get('violations'):
        with open(quality_dir / "compliance_violations.json", 'w') as f:
            json.dump(results['quality']['compliance']['violations'], f, indent=2)
        print(f"  - Compliance violations saved to: {quality_dir / 'compliance_violations.json'}")

    # Generate consolidated removal list
    print("\n  → Generating 'questions_to_remove.json'...")
    removal_candidates = {}
    
    # helper to add removal reason
    def add_removal(item, reason_prefix=""):
        q_id = item['id']
        if q_id not in removal_candidates:
            removal_candidates[q_id] = {
                'id': q_id,
                'text': item.get('text', ''),
                'reasons': []
            }
        
        reason = item.get('reason', '')
        if not reason and 'type' in item: # for compliance violations
             reason = f"{item['type']} (score: {item.get('score', 'N/A')})"
             
        full_reason = f"{reason_prefix}{reason}" if reason else reason_prefix.strip()
        if full_reason not in removal_candidates[q_id]['reasons']:
             removal_candidates[q_id]['reasons'].append(full_reason)

    # 1. Irrelevant data
    # if results['quality']['relevance'].get('irrelevant_data'):
    #     for item in results['quality']['relevance']['irrelevant_data']:
    #         add_removal({'id': item['id'], 'text': item['text']}, "Irrelevant content")

    # 2. Consistency issues
    if results['quality']['consistency'].get('issue'):
        for item in results['quality']['consistency']['issue']:
            add_removal(item)
            
    # 3. Suspicious correctness items
    if results['quality']['correctness'].get('suspicious_items'):
         for item in results['quality']['correctness']['suspicious_items']:
            add_removal(item, "Suspicious content: ")

    # 4. Duplicates
    if results['quality']['uniqueness'].get('to_remove'):
        for item in results['quality']['uniqueness']['to_remove']:
            add_removal(item)

    # 5. Compliance violations
    if results['quality']['compliance'].get('violations'):
        for item in results['quality']['compliance']['violations']:
            # compliance items might not have text, fetch if needed or use what's there
            # Since strict check_compliance doesn't return text in violation, we might miss it
            # But the id is key. We'll use empty text if missing.
            add_removal(item, "Compliance violation: ")

    # Save to file
    questions_to_remove = list(removal_candidates.values())
    if questions_to_remove:
        output_path = output_dir / "questions_to_remove.json"
        with open(output_path, 'w') as f:
            json.dump(questions_to_remove, f, indent=2)
        print(f"  - Questions to remove saved to: {output_path}")
        print(f"    Total candidates for removal: {len(questions_to_remove)}")
    else:
        print("  - No questions flagged for removal.")
    
    # Coverage artifacts
    coverage_dir = output_dir / "coverage"
    coverage_dir.mkdir(exist_ok=True)
    
    if results['coverage']['topic_distribution'].get('coverage_by_topic'):
        with open(coverage_dir / "topic_coverage_details.json", 'w') as f:
            json.dump(results['coverage']['topic_distribution']['coverage_by_topic'], f, indent=2)
        print(f"  - Topic coverage details saved to: {coverage_dir / 'topic_coverage_details.json'}")
    
    if results['coverage']['gap_analysis'].get('gap_details'):
        with open(coverage_dir / "topic_gaps.json", 'w') as f:
            json.dump(results['coverage']['gap_analysis']['gap_details'], f, indent=2)
        print(f"  - Topic gaps saved to: {coverage_dir / 'topic_gaps.json'}")
    
    # Quantity artifacts
    quantity_dir = output_dir / "quantity"
    quantity_dir.mkdir(exist_ok=True)
    
    if results['quantity']['by_category'].get('insufficient_categories'):
        with open(quantity_dir / "insufficient_categories.json", 'w') as f:
            json.dump(results['quantity']['by_category']['insufficient_categories'], f, indent=2)
        print(f"  - Insufficient categories saved to: {quantity_dir / 'insufficient_categories.json'}")
    
    if results['quantity']['interview_capacity'].get('details'):
        with open(quantity_dir / "interview_simulations.json", 'w') as f:
            json.dump(results['quantity']['interview_capacity']['details'], f, indent=2)
        print(f"  - Interview simulations saved to: {quantity_dir / 'interview_simulations.json'}")
    
    print(f"\n✅ All validation artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()


