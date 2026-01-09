# adaptive_sequencer.py

import numpy as np
from typing import List, Dict, Optional


class AdaptiveSequencer:
    """
    Implements simplified IRT-inspired adaptive sequencing for primary learners.

    Uses Bayesian ability estimation and pedagogically-informed question selection
    to personalize assessment experiences.
    """

    def __init__(self, learning_rate: float = 0.3,
                 initial_ability: float = 0.0,
                 ability_bounds: tuple = (-3.0, 3.0)):
        """
        Initialize adaptive sequencer.

        Args:
            learning_rate: α parameter controlling ability update speed (0-1)
            initial_ability: Starting θ value (typically 0 = average ability)
            ability_bounds: Min/max θ values to prevent extreme estimates
        """
        self.learning_rate = learning_rate
        self.initial_ability = initial_ability
        self.ability_bounds = ability_bounds

        # Session state
        self.current_ability = initial_ability
        self.ability_trajectory = [initial_ability]
        self.questions_presented = []
        self.responses = []
        self.consecutive_incorrect = 0

    def initialize_session(self, learner_id: str,
                           available_questions: List[Dict]):
        """
        Initialize a new adaptive assessment session.

        Args:
            learner_id: Unique identifier for learner
            available_questions: Pool of questions to select from
        """
        self.learner_id = learner_id
        self.available_questions = available_questions
        self.current_ability = self.initial_ability
        self.ability_trajectory = [self.initial_ability]
        self.questions_presented = []
        self.responses = []
        self.consecutive_incorrect = 0

    def select_next_question(self, curriculum_progress: Optional[List[str]] = None) -> Dict:
        """
        Select the next question to present based on current ability estimate.

        Args:
            curriculum_progress: List of learning outcome IDs already assessed

        Returns:
            Selected question dictionary
        """
        # Filter out already-presented questions
        candidate_questions = [
            q for q in self.available_questions
            if q['id'] not in [pq['id'] for pq in self.questions_presented]
        ]

        # Apply curriculum prerequisite constraints
        if curriculum_progress:
            candidate_questions = [
                q for q in candidate_questions
                if self._prerequisites_met(q, curriculum_progress)
            ]

        if not candidate_questions:
            return None  # No more questions available

        # Apply confidence boost if needed (2+ consecutive incorrect)
        target_difficulty = self.current_ability
        if self.consecutive_incorrect >= 2:
            target_difficulty = self.current_ability - 0.5  # Present easier question

        # Select question closest to target difficulty
        selected_question = min(
            candidate_questions,
            key=lambda q: abs(q['estimated_difficulty'] - target_difficulty)
        )

        # Apply content diversity (avoid consecutive questions from same topic)
        if self.questions_presented and len(candidate_questions) > 1:
            last_topic = self.questions_presented[-1].get('topic_area')
            if selected_question.get('topic_area') == last_topic:
                # Try to find alternative from different topic
                alternative_questions = [
                    q for q in candidate_questions
                    if q.get('topic_area') != last_topic
                ]
                if alternative_questions:
                    selected_question = min(
                        alternative_questions,
                        key=lambda q: abs(q['estimated_difficulty'] - target_difficulty)
                    )

        return selected_question

    def _prerequisites_met(self, question: Dict,
                           completed_outcomes: List[str]) -> bool:
        """
        Check if question's prerequisite learning outcomes have been assessed.
        """
        prerequisites = question.get('prerequisite_ids', [])
        return all(prereq in completed_outcomes for prereq in prerequisites)

    def update_ability(self, question: Dict, response: bool):
        """
        Update ability estimate based on question response using Bayesian update.

        Args:
            question: The presented question dictionary
            response: True if correct, False if incorrect
        """
        # Record response
        self.questions_presented.append(question)
        self.responses.append(response)

        # Update consecutive incorrect counter
        if response:
            self.consecutive_incorrect = 0
        else:
            self.consecutive_incorrect += 1

        # Calculate expected probability of correct response
        difficulty = question['estimated_difficulty']
        expected_prob = self._irt_probability(self.current_ability, difficulty)

        # Bayesian update
        ability_change = self.learning_rate * (float(response) - expected_prob)
        new_ability = self.current_ability + ability_change

        # Apply bounds
        new_ability = max(self.ability_bounds[0],
                          min(self.ability_bounds[1], new_ability))

        # Update state
        self.current_ability = new_ability
        self.ability_trajectory.append(new_ability)

    def _irt_probability(self, ability: float, difficulty: float) -> float:
        """
        Calculate probability of correct response using logistic IRT model.

        P(correct | θ, b) = 1 / (1 + exp(-(θ - b)))

        Args:
            ability: Learner ability (θ)
            difficulty: Question difficulty (b)

        Returns:
            Probability of correct response (0-1)
        """
        return 1.0 / (1.0 + np.exp(-(ability - difficulty)))

    def get_session_summary(self) -> Dict:
        """
        Generate summary statistics for the assessment session.

        Returns:
            Dict containing session metrics
        """
        if not self.responses:
            return {
                "total_questions": 0,
                "correct_responses": 0,
                "accuracy": 0.0,
                "final_ability": self.initial_ability,
                "ability_change": 0.0,
                "ability_trajectory": [self.initial_ability]
            }

        correct_count = sum(self.responses)
        total_count = len(self.responses)

        return {
            "learner_id": self.learner_id,
            "total_questions": total_count,
            "correct_responses": correct_count,
            "accuracy": correct_count / total_count,
            "initial_ability": self.initial_ability,
            "final_ability": self.current_ability,
            "ability_change": self.current_ability - self.initial_ability,
            "ability_trajectory": self.ability_trajectory,
            "questions_presented": [q['id'] for q in self.questions_presented],
            "responses": self.responses,
            "topic_coverage": self._calculate_topic_coverage()
        }

    def _calculate_topic_coverage(self) -> Dict[str, int]:
        """Calculate number of questions presented per topic area."""
        from collections import Counter
        topics = [q.get('topic_area', 'Unknown') for q in self.questions_presented]
        return dict(Counter(topics))

    def visualize_ability_trajectory(self, output_path: str):
        """
        Generate visualization of ability estimate evolution over session.

        Args:
            output_path: File path to save plot (e.g., 'trajectory.png')
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.ability_trajectory)), self.ability_trajectory,
                 marker='o', linestyle='-', linewidth=2, markersize=6)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5,
                    label='Average Ability')
        plt.xlabel('Question Number', fontsize=12)
        plt.ylabel('Ability Estimate (θ)', fontsize=12)
        plt.title('Learner Ability Trajectory During Adaptive Assessment',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
