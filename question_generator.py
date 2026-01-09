# question_generator.py
from typing import Dict, List, Optional
import json
import random

import textstat

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator


class QuestionSchema(BaseModel):
    """Pydantic model defining expected question structure."""
    question_stem: str = Field(..., min_length=10, max_length=500)
    correct_answer: str = Field(..., min_length=3, max_length=200)
    distractors: List[str] = Field(..., min_items=3, max_items=3)
    explanation: str = Field(..., min_length=20, max_length=1000)
    bloom_level: str = Field(..., pattern="^(Knowledge|Comprehension|Application|Analysis)$")
    estimated_difficulty: float = Field(..., ge=1.0, le=5.0)

    @validator('distractors')
    def check_distractor_lengths(cls, v):
        """Ensure distractors are similar length to avoid cueing."""
        lengths = [len(d) for d in v]
        if max(lengths) - min(lengths) > 50:
            raise ValueError("Distractors have inconsistent lengths")
        return v


class QuestionGenerator:
    """
    Generates curriculum-aligned MCQs using LLM prompting with
    few-shot learning and structured output parsing.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.8):
        # Slightly higher temperature helps diversity while still stable.
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            request_timeout=60
        )

        self.output_parser = PydanticOutputParser(pydantic_object=QuestionSchema)
        self.parser = self.output_parser

        # Load few-shot exemplars
        self.exemplars = self._load_exemplars()

        # Small pool of context variations (keeps novelty without changing requirements)
        self.scenario_seeds = [
            "at school",
            "at home",
            "during a class project",
            "when using a tablet",
            "when using a shared computer",
            "when playing an online game",
            "when using email",
            "when watching videos online",
        ]

    def _load_exemplars(self) -> List[Dict]:
        """Load manually authored exemplar questions for few-shot learning."""
        return [
            {
                "learning_outcome": "Students will be able to identify strong passwords.",
                "grade_level": "3",
                "bloom_level": "Knowledge",
                "difficulty": "2",
                "question": {
                    "question_stem": "Which password is the strongest?",
                    "correct_answer": "MyDog#2024!",
                    "distractors": ["password", "12345678", "mydog2024"],
                    "explanation": "MyDog#2024! is strongest because it combines uppercase letters, lowercase letters, numbers, and special characters, making it hard to guess.",
                    "bloom_level": "Knowledge",
                    "estimated_difficulty": 2.0
                }
            },
            {
                "learning_outcome": "Students will be able to explain why sharing personal information online can be risky.",
                "grade_level": "4",
                "bloom_level": "Comprehension",
                "difficulty": "3",
                "question": {
                    "question_stem": "Why should you not share your home address on social media?",
                    "correct_answer": "Strangers could find out where you live and visit without permission",
                    "distractors": [
                        "Your friends already know where you live",
                        "It takes too long to type your full address",
                        "Social media websites do not allow addresses"
                    ],
                    "explanation": "Sharing your home address online allows strangers to know where you live, which could be dangerous. Personal information should be kept private to protect your safety.",
                    "bloom_level": "Comprehension",
                    "estimated_difficulty": 3.0
                }
            }
        ]

    def generate_question(
        self,
        learning_outcome: Dict,
        target_difficulty: float = 3.0,
        max_attempts: int = 3,
        avoid_stems: Optional[List[str]] = None
    ) -> Optional[Dict]:

        avoid_stems = avoid_stems or []
        base_prompt = self._build_prompt(learning_outcome, target_difficulty, avoid_stems)

        prompt = base_prompt
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.llm.invoke(prompt)
                raw_output = getattr(response, "content", str(response))

                question = self.parser.parse(raw_output)

                if self._validate_question(question, learning_outcome):
                    return question.model_dump()
                else:
                    print(f"Generation attempt {attempt} failed validation.")
                    prompt = self._adjust_prompt_for_retry(base_prompt, attempt)

            except Exception as e:
                print(f"Generation attempt {attempt} failed: {e}")
                prompt = self._adjust_prompt_for_retry(base_prompt, attempt)

        return None

    def _build_prompt(self, learning_outcome: Dict, target_difficulty: float, avoid_stems: List[str]) -> str:
        """Construct prompt with system role, context, constraints, and examples."""

        grade = int(learning_outcome["grade_level"])
        scenario_hint = random.choice(self.scenario_seeds)

        system_message = (
            "You are an expert educational assessment designer specializing in "
            "digital literacy and cybersecurity education for primary school children in Qatar. "
            "Your questions must be age-appropriate, culturally sensitive, and aligned to curriculum outcomes."
        )

        context = f"""
Target Grade Level: Grade {grade} (ages {grade + 5}-{grade + 6})
Learning Outcome: {learning_outcome['outcome_text']}
Topic Area: {learning_outcome['topic_area']}
Bloom's Cognitive Level: {learning_outcome['bloom_level']}
Target Difficulty: {target_difficulty}/5.0
Scenario hint (use for variety): {scenario_hint}
"""

        constraints = """
CONSTRAINTS:
1. Use simple vocabulary appropriate for the target grade level
2. Keep question stem to 10–25 words maximum
3. Keep answer choices to 3–15 words each
4. Generate exactly three plausible but incorrect distractors
5. Avoid cultural insensitivity or stereotypes
6. Ensure distractors reflect common misconceptions
7. Provide a clear explanation (20–100 words)
8. Make the question clearly NEW and not a rephrase of any previous examples
"""

        avoid_text = ""
        if avoid_stems:
            avoid_text = (
                "\nDO NOT repeat or closely rephrase any of these previous question stems:\n" +
                "\n".join([f"- {s}" for s in avoid_stems])
            )

        examples_text = "\n\n".join([
            f"EXAMPLE {i + 1}:\n{json.dumps(ex['question'], indent=2)}"
            for i, ex in enumerate(self.exemplars)
        ])

        task = f"""
Generate one multiple-choice question.

OUTPUT FORMAT (JSON ONLY):
{self.output_parser.get_format_instructions()}

{avoid_text}
"""

        return f"{system_message}\n\n{context}\n\n{constraints}\n\n{examples_text}\n\n{task}"

    def _adjust_prompt_for_retry(self, original_prompt: str, attempt: int) -> str:
        """Adjust prompt for retry attempts to improve success rate & novelty."""
        adjustments = [
            "\n\nIMPORTANT: Make it clearly different from the examples and any previous stems (new scenario + new wording).",
            "\n\nIMPORTANT: Ensure distractors are plausible and similar length to the correct answer. Avoid obvious choices.",
            "\n\nIMPORTANT: Include at least one keyword from the learning outcome in the stem or options."
        ]
        idx = min(attempt - 1, len(adjustments) - 1)
        return original_prompt + adjustments[idx]

    def _validate_question(self, question: QuestionSchema, learning_outcome: Dict) -> bool:
        """Apply additional validation checks beyond schema validation."""

        # Bloom match
        if question.bloom_level != learning_outcome["bloom_level"]:
            return False

        # Keyword presence (if keywords exist)
        outcome_keywords = set(learning_outcome.get("keywords", []))
        question_text = (
            question.question_stem + " " +
            question.correct_answer + " " +
            " ".join(question.distractors)
        ).lower()

        if outcome_keywords:
            if not any(kw in question_text for kw in outcome_keywords):
                return False

        # Uniqueness of choices
        all_choices = [question.correct_answer] + question.distractors
        if len(set(all_choices)) != 4:
            return False

        # Readability check for primary level
        fk_grade = textstat.flesch_kincaid_grade(question.question_stem)
        if fk_grade > int(learning_outcome["grade_level"]) + 2:
            return False

        return True
