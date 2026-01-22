# scripts/parsers/wikipedia_concepts_collector.py

import wikipedia
import json
from pathlib import Path
from typing import List, Dict, Optional
import time
import requests

class WikipediaConceptCollector:
    """
    Collect ML/DS concept explanations from Wikipedia
    Uses local Ollama for enrichment
    """
    
    def __init__(self, use_llm_enrichment: bool = False, model: str = "llama3.1:8b"):
        """
        Args:
            use_llm_enrichment: Use local LLM to generate simple_explanation, examples, etc.
            model: Ollama model to use (e.g., llama3.1:8b, ministral-3:14b)
        """
        self.use_llm_enrichment = use_llm_enrichment
        self.model = model
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        
        # Test Ollama connection
        if use_llm_enrichment:
            if self._test_ollama():
                print(f"✓ Connected to Ollama (model: {model})")
            else:
                print("✗ Ollama not running. Install: https://ollama.ai")
                print("  Then run: ollama pull llama3.1:8b")
                self.use_llm_enrichment = False
        
        # Category mapping
        self.category_keywords = {
            'optimization': ['gradient', 'descent', 'optimizer', 'learning rate', 'momentum', 'adam'],
            'algorithms': ['algorithm', 'search', 'sort', 'tree', 'forest', 'svm', 'knn'],
            'architectures': ['neural network', 'cnn', 'rnn', 'lstm', 'transformer', 'architecture'],
            'techniques': ['regularization', 'dropout', 'normalization', 'augmentation', 'ensemble'],
            'evaluation': ['metric', 'accuracy', 'precision', 'recall', 'f1', 'roc', 'auc'],
            'fundamentals': ['supervised', 'unsupervised', 'reinforcement', 'learning', 'training'],
            'preprocessing': ['normalization', 'scaling', 'encoding', 'feature', 'pca'],
            'statistics': ['probability', 'distribution', 'variance', 'mean', 'statistics']
        }
    
    def _test_ollama(self) -> bool:
        """Test if Ollama is running"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": "test",
                    "stream": False
                },
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def collect_concepts(self, concept_list: List[str]) -> List[Dict]:
        """
        Collect concepts from Wikipedia
        
        Args:
            concept_list: List of concept names to fetch
        
        Returns:
            List of structured concept dictionaries
        """
        concepts = []
        
        print(f"Collecting {len(concept_list)} concepts from Wikipedia...")
        
        for i, concept_name in enumerate(concept_list):
            print(f"\n[{i+1}/{len(concept_list)}] Fetching: {concept_name}")
            
            try:
                concept_data = self._fetch_concept(concept_name)
                
                if concept_data:
                    concepts.append(concept_data)
                    print(f"  ✓ Success")
                else:
                    print(f"  ✗ Failed")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        print(f"\n✓ Successfully collected {len(concepts)} concepts")
        return concepts
    
    def _fetch_concept(self, concept_name: str) -> Optional[Dict]:
        """Fetch single concept from Wikipedia"""
        
        try:
            # Search for the page
            page = wikipedia.page(concept_name, auto_suggest=True)
            
            # Get content
            summary = page.summary
            full_content = page.content[:3000]  # First 3000 chars
            
            # Basic structure
            concept_data = {
                'id': self._generate_id(concept_name),
                'concept_name': page.title,
                'explanation': summary,
                'category': self._categorize_concept(concept_name, summary),
                'source': 'wikipedia',
                'url': page.url
            }
            
            # Optional: Enrich with local LLM
            if self.use_llm_enrichment:
                print("    Enriching with LLM...", end='', flush=True)
                enriched = self._enrich_with_ollama(concept_name, summary)
                concept_data.update(enriched)
                print(" Done")
            else:
                # Add placeholder fields
                concept_data.update({
                    'simple_explanation': '',
                    'examples': [],
                    'related_concepts': [],
                    'common_misconceptions': [],
                    'when_to_use': ''
                })
            
            return concept_data
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Try first option from disambiguation
            try:
                page = wikipedia.page(e.options[0])
                return self._fetch_concept(e.options[0])
            except:
                return None
                
        except wikipedia.exceptions.PageError:
            return None
        
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def _generate_id(self, concept_name: str) -> str:
        """Generate ID from concept name"""
        return 'concept_' + concept_name.lower().replace(' ', '_').replace('-', '_')
    
    def _categorize_concept(self, concept_name: str, explanation: str) -> str:
        """Categorize concept based on keywords"""
        text = (concept_name + ' ' + explanation).lower()
        
        scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[category] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def _enrich_with_ollama(self, concept_name: str, explanation: str) -> Dict:
        """Use local Ollama LLM to generate additional fields"""
        
        prompt = f"""You are an ML/DS educator. Given this concept, provide structured enrichment.

Concept: {concept_name}
Explanation: {explanation[:500]}

Generate:
1. Simple explanation (ELI5, 1-2 sentences max)
2. 2-3 practical examples where this is used
3. 3-5 related concepts (just names)
4. 2-3 common misconceptions
5. When to use this (1 sentence)

Output ONLY valid JSON, no other text:
{{
    "simple_explanation": "...",
    "examples": ["example1", "example2"],
    "related_concepts": ["concept1", "concept2", "concept3"],
    "common_misconceptions": ["misconception1", "misconception2"],
    "when_to_use": "..."
}}"""
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 400  # Max tokens
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()['response']
                
                # Parse JSON from response
                import re
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    enriched = json.loads(json_match.group())
                    
                    # Validate structure
                    required_keys = ['simple_explanation', 'examples', 'related_concepts', 
                                    'common_misconceptions', 'when_to_use']
                    if all(key in enriched for key in required_keys):
                        return enriched
            
        except Exception as e:
            print(f"\n    LLM enrichment failed: {e}")
        
        # Return empty fields if LLM fails
        return {
            'simple_explanation': '',
            'examples': [],
            'related_concepts': [],
            'common_misconceptions': [],
            'when_to_use': ''
        }


def get_ml_concept_list() -> List[str]:
    """
    Predefined list of important ML/DS concepts
    """
    # return [
    #     # Fundamentals (15)
    #     'supervised learning',
    #     'unsupervised learning',
    #     'reinforcement learning',
    #     'overfitting',
    #     'underfitting',
    #     'bias-variance tradeoff',
    #     'cross-validation',
    #     'regularization',
    #     'feature engineering',
    #     'hyperparameter tuning',
    #     'training set',
    #     'validation set',
    #     'test set',
    #     'batch size',
    #     'epoch',
        
    #     # Optimization (10)
    #     'gradient descent',
    #     'stochastic gradient descent',
    #     'learning rate',
    #     'momentum',
    #     'Adam optimizer',
    #     'batch normalization',
    #     'learning rate schedule',
    #     'gradient clipping',
    #     'weight initialization',
    #     'loss function',
        
    #     # Classical ML Algorithms (12)
    #     'linear regression',
    #     'logistic regression',
    #     'decision tree',
    #     'random forest',
    #     'support vector machine',
    #     'k-nearest neighbors',
    #     'k-means clustering',
    #     'principal component analysis',
    #     'naive bayes',
    #     'ensemble learning',
    #     'bagging',
    #     'boosting',
        
    #     # Neural Networks (15)
    #     'artificial neural network',
    #     'backpropagation',
    #     'convolutional neural network',
    #     'recurrent neural network',
    #     'long short-term memory',
    #     'transformer',
    #     'attention mechanism',
    #     'activation function',
    #     'relu',
    #     'sigmoid',
    #     'softmax',
    #     'dropout',
    #     'early stopping',
    #     'transfer learning',
    #     'fine-tuning',
        
    #     # Evaluation Metrics (8)
    #     'confusion matrix',
    #     'precision and recall',
    #     'F1 score',
    #     'ROC curve',
    #     'AUC',
    #     'mean squared error',
    #     'mean absolute error',
    #     'R-squared',
        
    #     # Preprocessing (8)
    #     'feature scaling',
    #     'one-hot encoding',
    #     'normalization',
    #     'standardization',
    #     'data augmentation',
    #     'imbalanced data',
    #     'missing data',
    #     'outlier detection',
        
    #     # Advanced Topics (12)
    #     'gradient boosting',
    #     'word2vec',
    #     'embedding',
    #     'autoencoder',
    #     'generative adversarial network',
    #     'variational autoencoder',
    #     'batch normalization',
    #     'layer normalization',
    #     'residual connection',
    #     'skip connection',
    #     'curriculum learning',
    #     'meta-learning',
        
    #     # Statistics & Math (10)
    #     'probability distribution',
    #     'normal distribution',
    #     'maximum likelihood estimation',
    #     'bayes theorem',
    #     'covariance',
    #     'correlation',
    #     'hypothesis testing',
    #     'p-value',
    #     'confidence interval',
    #     'variance'
    # ]
    return [
        "Vanishing_gradient_problem",
        "Decision_tree",
        "Transformer (deep learning)",
        "Softmax function",
        "F-score",
        "AUC",
        "Residual_neural_network",
        "p-value",
        "Self-Attention Mechanism",
        "Multi-Head Attention",
        "Positional Encoding",
        "Encoder-Decoder Architecture",
        "Feed-Forward Networks in Transformers",
        "Residual Connections",
        "Layer Norm (Pre-Norm vs. Post-Norm)",
        'Pre-training (Next Token Prediction/Causal LM)',
 ' Masked Language Modeling (MLM)',
 ' Fine-Tuning (SFT - Supervised Fine-Tuning)',
 ' RLHF (Reinforcement Learning from Human Feedback)',
 ' DPO (Direct Preference Optimization)',
 ' PPO (Proximal Policy Optimization)',
 'LoRA (Low-Rank Adaptation)',
 ' QLoRA',
 ' Quantization (4-bit',
 ' 8-bit',
 ' FP16 vs BF16)',
 ' Adapters',
 ' Prompt Tuning',
 ' Prefix Tuning',
 'Temperature',
 ' Top-K Sampling',
 ' Top-P (Nucleus) Sampling',
 ' Beam Search',
 ' Greedy Decoding',
 ' KV Cache',
 ' Context Window',
 'Vector Embeddings',
 ' Semantic Search',
 ' Cosine Similarity',
 ' HNSW Index',
 ' Chunking Strategies (Fixed-size',
 ' Semantic)',
 ' Re-ranking (Cross-Encoders)',
 ' Dense vs. Sparse Retrieval (BM25 vs. Embeddings)'
    ]


def main():
    """Main execution"""
    
    # Configuration
    USE_LLM_ENRICHMENT = True  # Set to True to use local LLM
    MODEL = "llama3.1:8b"  # Options: llama3.1:8b, ministral-3:14b, etc.
    
    collector = WikipediaConceptCollector(
        use_llm_enrichment=USE_LLM_ENRICHMENT,
        model=MODEL
    )
    
    # Get concept list
    concepts_to_collect = get_ml_concept_list()
    
    print(f"\nWill collect {len(concepts_to_collect)} concepts")
    print(f"LLM Enrichment: {'Enabled' if USE_LLM_ENRICHMENT else 'Disabled'}")
    if USE_LLM_ENRICHMENT:
        print(f"Model: {MODEL}")
    print("")
    
    # Collect
    concepts = collector.collect_concepts(concepts_to_collect)
    
    # Save to file
    output_dir = Path("./data/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "wikipedia_ml_concepts_part2.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(concepts, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(concepts)} concepts to {output_file}")
    
    # Display sample
    print("\n" + "="*60)
    print("SAMPLE CONCEPTS")
    print("="*60)
    
    for concept in concepts[:2]:
        print(f"\nConcept: {concept['concept_name']}")
        print(f"Category: {concept['category']}")
        print(f"Explanation: {concept['explanation'][:150]}...")
        if concept.get('simple_explanation'):
            print(f"Simple: {concept['simple_explanation']}")
        if concept.get('examples'):
            print(f"Examples: {', '.join(concept['examples'][:2])}")
        print("-" * 60)
    
    # Statistics
    from collections import Counter
    
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    
    category_counts = Counter(c['category'] for c in concepts)
    print("\nCategory Distribution:")
    for category, count in category_counts.most_common():
        print(f"  {category}: {count}")
    
    enriched_count = sum(1 for c in concepts if c.get('simple_explanation'))
    print(f"\nEnriched with LLM: {enriched_count}/{len(concepts)}")


if __name__ == "__main__":
    main()