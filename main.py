import os
from curriculum_mapper import CurriculumMapper
from question_generator import QuestionGenerator
from difficulty_estimator import DifficultyEstimator
from adaptive_engine import AdaptiveSequencer


def main():
    print("\n=== AI-BASED QUESTION GENERATION SYSTEM ===\n")

    # -------------------------------------------------------
    # 1. Load curriculum outcomes
    # -------------------------------------------------------
    mapper = CurriculumMapper()

    # Example outcomes (you can replace with your JSON file later)
    outcomes = [
        {
            "description": "Students will be able to identify strong and weak passwords.",
            "grade_level": 6,
            "topic": "Cybersecurity"
        },
        {
            "description": "Students will classify different types of digital threats.",
            "grade_level": 6,
            "topic": "Cybersecurity"
        }
    ]

    mapped_outcomes = mapper.map_outcomes(outcomes)
    print("Loaded & mapped curriculum outcomes.")
    print(mapped_outcomes)

    # -------------------------------------------------------
    # 2. Generate a question using OpenAI
    # -------------------------------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Set your OPENAI_API_KEY environment variable.")

    qg = QuestionGenerator(api_key=api_key)

    print("\nGenerating a question...\n")
    question_item = qg.generate_question(mapped_outcomes[0])
    print("Generated Question:")
    print(question_item)

    if question_item is None:
        print("\n❌ Question generation failed after multiple attempts. Exiting.")
        return

    # -------------------------------------------------------
    # 3. Estimate difficulty
    # -------------------------------------------------------
    diff = DifficultyEstimator()
    difficulty_label = diff.estimate_difficulty(question_item)

    print("\nEstimated Difficulty:", difficulty_label)

    # -------------------------------------------------------
    # 4. Adaptive Sequencing
    # -------------------------------------------------------
    sequencer = AdaptiveSequencer()

    # Simulate a student answering correctly
    next_level = sequencer.next_question_difficulty(
        difficulty_label=difficulty_label,
        correct=True
    )

    print("\nNext recommended difficulty level:", next_level)

    print("\n=== SYSTEM RUN SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
