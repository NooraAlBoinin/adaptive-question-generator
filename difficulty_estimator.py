# difficulty_estimator.py

from typing import List, Dict
import textstat
from sentence_transformers import SentenceTransformer
import numpy as np


class DifficultyEstimator:
    """
    Estimates question difficulty using hybrid approach combining
    linguistic metrics, cognitive indicators, and distractor analysis.
    """

    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Bloom's level difficulty weights (empirically determined)
        self.bloom_weights = {
            "Knowledge": 1.0,
            "Comprehension": 2.0,
            "Application": 3.5,
            "Analysis": 4.5
        }

        # Component weights for composite score
        self.linguistic_weight = 0.3
        self.cognitive_weight = 0.5
        self.distractor_weight = 0.2

    def estimate_difficulty(self, question: Dict) -> Dict[str, float]:
        """
        Estimate question difficulty across multiple dimensions.

        Args:
            question: Dict containing question_stem, correct_answer, distractors, bloom_level

        Returns:
            Dict with linguistic_score, cognitive_score, distractor_score, composite_difficulty
        """
        # Calculate component scores
        linguistic_score = self._calculate_linguistic_complexity(question)
        cognitive_score = self._calculate_cognitive_demand(question)
        distractor_score = self._calculate_distractor_plausibility(question)

        # Compute weighted composite
        composite = (
                self.linguistic_weight * linguistic_score +
                self.cognitive_weight * cognitive_score +
                self.distractor_weight * distractor_score
        )

        # Normalize to 1-5 scale
        composite_normalized = max(1.0, min(5.0, composite))

        return {
            "linguistic_score": round(linguistic_score, 2),
            "cognitive_score": round(cognitive_score, 2),
            "distractor_score": round(distractor_score, 2),
            "composite_difficulty": round(composite_normalized, 2)
        }

    def _calculate_linguistic_complexity(self, question: Dict) -> float:
        """
        Calculate linguistic complexity based on readability metrics.

        Components:
        - Flesch-Kincaid Grade Level (reading difficulty)
        - Vocabulary rarity (proportion of uncommon words)
        - Syntactic complexity (sentence structure indicators)
        """
        text = question['question_stem']

        # Flesch-Kincaid Grade Level (normalized to 1-5)
        fk_grade = textstat.flesch_kincaid_grade(text)
        fk_normalized = max(1.0, min(5.0, fk_grade / 3.0))  # Divide by 3 to map typical range 3-15 to 1-5

        # Vocabulary rarity using word frequency
        words = text.lower().split()
        # Load word frequency lexicon (simplified example)
        common_words = set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'you',
                            'your', 'should', 'what', 'which', 'how', 'why'])
        rare_word_proportion = sum(1 for w in words if w not in common_words) / len(words)
        rarity_score = rare_word_proportion * 5.0  # Scale to 1-5

        # Syntactic complexity (simple heuristic: subordinate clauses, passive voice)
        complexity_indicators = [
            'because', 'although', 'however', 'therefore', 'unless',
            'was', 'were', 'been'  # Passive voice indicators
        ]
        syntax_score = sum(1 for ind in complexity_indicators if ind in text.lower())
        syntax_normalized = min(5.0, 1.0 + syntax_score * 0.5)

        # Average the three components
        linguistic_score = (fk_normalized + rarity_score + syntax_normalized) / 3.0

        return linguistic_score

    def _calculate_cognitive_demand(self, question: Dict) -> float:
        """
        Cognitive demand estimation using Bloom's level and
        lightweight linguistic heuristics (no external NLP models).
        """
        bloom_level = question.get("bloom_level", "Comprehension")
        bloom_score = self.bloom_weights.get(bloom_level, 2.0)

        text = question["question_stem"]
        words = text.split()

        avg_word_len = sum(len(w) for w in words) / max(1, len(words))
        sentence_len = len(words)

        # Heuristic concept density
        concept_density = (
                0.5 * min(5.0, sentence_len / 6.0) +
                0.5 * min(5.0, avg_word_len)
        )

        return (bloom_score + concept_density) / 2.0

    def _calculate_distractor_plausibility(self, question: Dict) -> float:
        """
        Calculate distractor plausibility using semantic similarity.

        Higher similarity between distractors and correct answer indicates
        more plausible (and thus more difficult) distractors.
        """
        correct_answer = question['correct_answer']
        distractors = question['distractors']

        # Compute embeddings
        correct_embedding = self.embedding_model.encode(correct_answer)
        distractor_embeddings = self.embedding_model.encode(distractors)

        # Calculate cosine similarity between correct answer and each distractor
        similarities = []
        for dist_emb in distractor_embeddings:
            similarity = np.dot(correct_embedding, dist_emb) / (
                    np.linalg.norm(correct_embedding) * np.linalg.norm(dist_emb)
            )
            similarities.append(similarity)

        # Average similarity (higher = more plausible distractors = harder question)
        avg_similarity = np.mean(similarities)

        # Map similarity (typically 0.3-0.8 range) to 1-5 scale
        plausibility_score = 1.0 + (avg_similarity * 5.0)
        plausibility_score = max(1.0, min(5.0, plausibility_score))

        return plausibility_score

    def calibrate_weights(self, questions: List[Dict],
                          teacher_ratings: List[float]):
        """
        Calibrate component weights based on teacher difficulty ratings.

        Uses linear regression to find optimal weights that minimize
        difference between estimated and teacher-rated difficulty.

        Args:
            questions: List of question dictionaries with component scores
            teacher_ratings: List of teacher difficulty ratings (1-5 scale)
        """
        from sklearn.linear_model import LinearRegression

        # Prepare feature matrix
        X = np.array([
            [q['linguistic_score'], q['cognitive_score'], q['distractor_score']]
            for q in questions
        ])
        y = np.array(teacher_ratings)

        # Fit linear regression
        model = LinearRegression(fit_intercept=False, positive=True)
        model.fit(X, y)

        # Update weights (normalize to sum to 1.0)
        weights = model.coef_
        weights_normalized = weights / weights.sum()

        self.linguistic_weight = weights_normalized[0]
        self.cognitive_weight = weights_normalized[1]
        self.distractor_weight = weights_normalized[2]

        print(f"Calibrated weights: Linguistic={self.linguistic_weight:.2f}, "
              f"Cognitive={self.cognitive_weight:.2f}, "
              f"Distractor={self.distractor_weight:.2f}")
