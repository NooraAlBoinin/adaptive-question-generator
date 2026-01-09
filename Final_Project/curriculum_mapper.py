import json
import re
from collections import Counter
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import numpy as np


class CurriculumMapper:
    """
    Extracts learning outcomes from curriculum text and structures them
    for question generation and adaptive sequencing.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        # Sentence-transformer for semantic embeddings
        self.model = SentenceTransformer(embedding_model)

        # Simple Bloom's taxonomy keyword mapping
        self.bloom_keywords = {
            "Knowledge": ["define", "list", "name", "identify", "recall", "state"],
            "Comprehension": ["explain", "describe", "summarise", "classify", "discuss"],
            "Application": ["apply", "use", "solve", "demonstrate", "implement"],
            "Analysis": ["compare", "contrast", "analyse", "differentiate", "examine"],
        }

    # ------------------------------------------------------------------
    # Main extraction pipeline
    # ------------------------------------------------------------------
    def extract_learning_outcomes(
        self,
        pdf_text: str,
        grade_level: int,
        curriculum_source: str,
    ) -> List[Dict]:
        """
        Extract learning outcomes from curriculum text.

        Args:
            pdf_text: Extracted text from curriculum PDF.
            grade_level: Target grade level (1–6).
            curriculum_source: Identifier (e.g. "MOE_ICT_2024", "NCSA_Cyber_2023").

        Returns:
            List of structured learning outcome dictionaries.
        """
        outcomes: List[Dict] = []

        # Pattern matching for common outcome statement formats
        patterns = [
            r"students? will be able to (.+?)(?:\.|;|\n)",
            r"learners? will (.+?)(?:\.|;|\n)",
            r"by the end of this unit, students? will (.+?)(?:\.|;|\n)",
            r"learning outcome[s]?:\s*(.+?)(?:\.|;|\n)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, pdf_text, re.IGNORECASE)
            for match in matches:
                outcome_text = match.group(1).strip()

                # Skip if outcome too short or too long (likely extraction noise)
                if len(outcome_text) < 20 or len(outcome_text) > 300:
                    continue

                # Classify Bloom's level
                bloom_level = self._classify_bloom_level(outcome_text)

                # Extract keywords (no spaCy – lightweight heuristic)
                keywords = self._extract_keywords(outcome_text)

                # Determine topic area based on keyword analysis
                topic_area = self._determine_topic_area(outcome_text, keywords)

                outcome_id = f"{curriculum_source}_G{grade_level}_{len(outcomes) + 1}"

                outcome = {
                    "id": outcome_id,
                    "curriculum_id": f"{curriculum_source}_G{grade_level}",
                    "grade_level": grade_level,
                    "topic_area": topic_area,
                    "outcome_text": outcome_text,
                    "bloom_level": bloom_level,
                    "keywords": keywords,
                    "prerequisite_ids": [],
                    "embedding": self.model.encode(outcome_text).tolist(),
                }

                outcomes.append(outcome)

        return outcomes

    def map_outcomes(self, outcomes: List[Dict], curriculum_source: str = "Manual") -> List[Dict]:
        """
        Take a simple list of outcome dictionaries (with description/grade/topic)
        and enrich them with Bloom level, keywords, topic area, and embeddings.

        This is used in main.py when we manually define outcomes in Python.
        """
        mapped: List[Dict] = []

        for idx, raw in enumerate(outcomes, start=1):
            # Accept either "description" or "outcome_text"
            text = (raw.get("outcome_text") or raw.get("description") or "").strip()
            if not text:
                continue

            grade = raw.get("grade_level", 6)

            # Classify Bloom level
            bloom_level = self._classify_bloom_level(text)

            # Extract keywords
            keywords = self._extract_keywords(text)

            # Determine topic area (respect explicit topic if already provided)
            topic_area = raw.get("topic") or self._determine_topic_area(text, keywords)

            outcome_id = f"{curriculum_source}_G{grade}_{idx}"

            mapped.append(
                {
                    "id": outcome_id,
                    "curriculum_id": f"{curriculum_source}_G{grade}",
                    "grade_level": grade,
                    "topic_area": topic_area,
                    "outcome_text": text,
                    "bloom_level": bloom_level,
                    "keywords": keywords,
                    "prerequisite_ids": [],
                    "embedding": self.model.encode(text).tolist(),
                }
            )

        return mapped





    # ------------------------------------------------------------------
    # Bloom classification
    # ------------------------------------------------------------------
    def _classify_bloom_level(self, outcome_text: str) -> str:
        """Classify cognitive level based on action verbs."""
        outcome_lower = outcome_text.lower()

        # Check for Bloom keywords in order (higher levels override lower)
        for level in ["Knowledge", "Comprehension", "Application", "Analysis"]:
            for keyword in self.bloom_keywords[level]:
                if keyword in outcome_lower:
                    return level

        # Default to Comprehension if no clear indicator
        return "Comprehension"

    # ------------------------------------------------------------------
    # Keyword extraction (no spaCy)
    # ------------------------------------------------------------------
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract domain-specific keywords using simple tokenisation and frequency.
        This avoids heavy NLP dependencies like spaCy.
        """
        # Basic tokenisation: alphabetic tokens of length >= 4
        tokens = re.findall(r"[A-Za-z]{4,}", text.lower())

        # Minimal stopword list; extend if needed
        stopwords = {
            "students", "student", "learners", "learning",
            "ability", "abilities", "skills",
            "will", "able", "understand", "understanding",
            "know", "knows", "using", "into", "from", "with",
            "this", "that", "their", "these", "those", "about",
            "them", "they", "your", "have", "been", "such", "through",
        }

        filtered = [t for t in tokens if t not in stopwords]
        if not filtered:
            return []

        counts = Counter(filtered)
        keywords = [word for word, _ in counts.most_common(top_n)]
        return keywords

    # ------------------------------------------------------------------
    # Topic area detection
    # ------------------------------------------------------------------
    def _determine_topic_area(self, outcome_text: str, keywords: List[str]) -> str:
        """Determine topic area based on keyword analysis."""
        topic_keywords = {
            "Online Safety": [
                "safety", "safe", "danger", "risk", "protect", "secure", "privacy"
            ],
            "Password Security": [
                "password", "authentication", "login", "credential", "secure password"
            ],
            "Phishing Recognition": [
                "phishing", "scam", "fraud", "fake", "suspicious", "email"
            ],
            "Digital Footprint": [
                "footprint", "trace", "data", "information", "sharing", "post"
            ],
            "Digital Citizenship": [
                "citizenship", "responsible", "ethical", "respectful", "rights"
            ],
            "Cybersecurity Basics": [
                "cybersecurity", "cyber", "threat", "attack", "malware", "virus"
            ],
        }

        outcome_lower = outcome_text.lower()
        keywords_joined = " ".join(keywords).lower()

        topic_scores: Dict[str, int] = {}
        for topic, topic_kws in topic_keywords.items():
            score = sum(
                1 for kw in topic_kws
                if kw in outcome_lower or kw in keywords_joined
            )
            topic_scores[topic] = score

        best_topic = max(topic_scores, key=topic_scores.get)
        return best_topic if topic_scores[best_topic] > 0 else "Digital Literacy"

    # ------------------------------------------------------------------
    # Prerequisite graph
    # ------------------------------------------------------------------
    def build_prerequisite_graph(self, outcomes: List[Dict]) -> List[Dict]:
        """
        Identify prerequisite relationships between learning outcomes
        using semantic similarity and grade level progression.
        """
        # Sort outcomes by grade level
        outcomes_sorted = sorted(outcomes, key=lambda x: x["grade_level"])

        for i, outcome in enumerate(outcomes_sorted):
            prerequisites = []

            # Only check outcomes from earlier grades
            for potential_prereq in outcomes_sorted[:i]:
                if potential_prereq["grade_level"] >= outcome["grade_level"]:
                    continue

                # Check semantic similarity
                similarity = self._compute_similarity(
                    outcome["embedding"], potential_prereq["embedding"]
                )

                # If highly similar and from earlier grade, likely prerequisite
                if similarity > 0.7:
                    prerequisites.append(potential_prereq["id"])

            outcome["prerequisite_ids"] = prerequisites

        return outcomes_sorted

    # ------------------------------------------------------------------
    # Similarity + export
    # ------------------------------------------------------------------
    def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between embeddings."""
        vec1 = np.array(embedding1, dtype=float)
        vec2 = np.array(embedding2, dtype=float)

        denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denom == 0:
            return 0.0

        cosine_sim = float(np.dot(vec1, vec2) / denom)
        return cosine_sim

    def export_to_json(self, outcomes: List[Dict], output_path: str) -> None:
        """Export structured curriculum to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "curriculum_version": "1.0",
                    "total_outcomes": len(outcomes),
                    "grade_levels": sorted({o["grade_level"] for o in outcomes}),
                    "outcomes": outcomes,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
