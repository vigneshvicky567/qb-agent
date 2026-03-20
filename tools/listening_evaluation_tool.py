from pydantic import BaseModel
from typing import List
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
import os 
from dotenv import load_dotenv

load_dotenv()

class QAPair(TypedDict):
    question: str
    candidate_answer: str
    correct_answer: str

class ListeningEvaluationResult(BaseModel):
    comprehension_score: float  # 0-100
    correct_answers: int
    total_questions: int
    accuracy_percentage: float
    understanding_level: str  # "poor", "fair", "good", "excellent"
    missed_key_points: List[str]
    feedback: str
    recommendations: str

class ListeningEvaluationOutput(BaseModel):
    evaluation: ListeningEvaluationResult

def evaluate_listening(passage: str, questions_and_answers: List[QAPair]) -> str:
    """
    Evaluate listening comprehension based on the candidate's answers to questions about a passage.
    
    Args:
        passage (str): The passage that was listened to.
        questions_and_answers (List[dict]): List of dicts with format:
            {
                "question": "What is...",
                "candidate_answer": "The answer given",
                "correct_answer": "The correct answer"
            }
    
    Returns:
        str: JSON string with listening evaluation results.
    """
    print(f"TOOL: Evaluating listening comprehension")
    
    # Calculate basic metrics
    correct_count = 0
    for qa in questions_and_answers:
        # Simple comparison - can be enhanced with semantic similarity
        if qa.get("candidate_answer", "").lower().strip() == qa.get("correct_answer", "").lower().strip():
            correct_count += 1
    
    total_questions = len(questions_and_answers)
    accuracy_percentage = round((correct_count / total_questions * 100), 2) if total_questions > 0 else 0
    
    max_tokens = int(os.getenv("MAX_TOKENS_FROM_TOOLS", 1500))
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=max_tokens, temperature=0.3)
    structured_llm = llm.with_structured_output(ListeningEvaluationOutput, method="function_calling")
    
    # Format Q&A for prompt
    qa_text = "\n".join([
        f"Q: {qa['question']}\n   Candidate Answer: {qa['candidate_answer']}\n   Correct Answer: {qa['correct_answer']}\n"
        for qa in questions_and_answers
    ])
    
    prompt = f"""You are an expert English listening comprehension evaluator.

      Evaluate the candidate's listening comprehension based on their answers to questions about the passage.

      PASSAGE:
      {passage}

      QUESTIONS AND ANSWERS:
      {qa_text}

      ACCURACY METRICS:
      - Correct Answers: {correct_count} out of {total_questions}
      - Accuracy: {accuracy_percentage}%

      Provide evaluation in JSON format with:
      1. comprehension_score (0-100): Overall comprehension level
      2. correct_answers: {correct_count} (calculated value)
      3. total_questions: {total_questions} (calculated value)
      4. accuracy_percentage: {accuracy_percentage} (calculated value)
      5. understanding_level: Categorize as "poor" (<50%), "fair" (50-70%), "good" (70-85%), "excellent" (>85%)
      6. missed_key_points: List of main ideas from the passage that the candidate failed to grasp
      7. feedback: Specific observations about comprehension strengths and weaknesses
      8. recommendations: Actionable suggestions for improvement

      Consider semantic understanding, not just exact word matching, in your detailed analysis."""

    result: ListeningEvaluationOutput = structured_llm.invoke(prompt)
    # Override calculated fields to ensure consistency with pre-computed values
    result.evaluation.correct_answers = correct_count
    result.evaluation.total_questions = total_questions
    result.evaluation.accuracy_percentage = accuracy_percentage
    return result.model_dump_json()
