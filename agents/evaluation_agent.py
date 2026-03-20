from pydantic import BaseModel, Field
from typing import Optional, Any, List
from dotenv import load_dotenv
from langchain.agents import create_agent
from langsmith import traceable
from .prompts import EVALUATION_SYSTEM_PROMPT
from tools.grammar_evaluation_tool import evaluate_grammar
from tools.reading_evaluation_tool import evaluate_reading
from tools.listening_evaluation_tool import evaluate_listening
from langchain.agents.middleware import ToolCallLimitMiddleware
import json

load_dotenv()

grammar_eval_limiter = ToolCallLimitMiddleware(tool_name="evaluate_grammar", run_limit=2, exit_behavior="error")
reading_eval_limiter = ToolCallLimitMiddleware(tool_name="evaluate_reading", run_limit=2, exit_behavior="error")
listening_eval_limiter = ToolCallLimitMiddleware(tool_name="evaluate_listening", run_limit=2, exit_behavior="error")

class EvaluationOutput(BaseModel):
    evaluation_type: str
    score: Optional[float] = None
    feedback: str
    detailed_results: dict = Field(default_factory=dict)


def build_evaluation_agent():
    return create_agent(
        model="gpt-4o-mini",
        tools=[evaluate_grammar, evaluate_reading, evaluate_listening],
        middleware=[grammar_eval_limiter, reading_eval_limiter, listening_eval_limiter],
        system_prompt=EVALUATION_SYSTEM_PROMPT,
        response_format=EvaluationOutput
    )


@traceable
async def run_evaluation_agent(evaluation_type: str, **kwargs):
    """
    Run the evaluation agent for the specified evaluation type.
    
    Args:
        evaluation_type: "grammar", "reading", or "listening"
        **kwargs: Additional parameters specific to evaluation type:
            - Grammar: original_passage, transcribed_text
            - Reading: original_passage, transcribed_text, audio_duration_seconds
            - Listening: passage, questions_and_answers (list of dicts with question, candidate_answer, correct_answer)
    Returns:
        dict: Evaluation results
    """
    print(f"EVALUATION_AGENT: Running agent for evaluation type: {evaluation_type}")
    
    agent = build_evaluation_agent()
    print("EVALUATION_AGENT: Agent built successfully, invoking now...")
    
    # Build the user message based on evaluation type
    if evaluation_type == "grammar":
        user_message = f"""Please evaluate the candidate's grammar skills.

        Original Passage:
        {kwargs.get('original_passage', '')}

        Candidate's Transcribed Text:
        {kwargs.get('transcribed_text', '')}

        Use the evaluate_grammar tool to perform the assessment."""

    elif evaluation_type == "reading":
        user_message = f"""Please evaluate the candidate's reading skills.

        Original Passage:
        {kwargs.get('original_passage', '')}

        Candidate's Transcribed Text:
        {kwargs.get('transcribed_text', '')}

        Audio Duration: {kwargs.get('audio_duration_seconds', 0)} seconds

        Use the evaluate_reading tool to assess reading speed, fluency, and pronunciation."""

    elif evaluation_type == "listening":
        qa_text = ""
        for qa in kwargs.get('questions_and_answers', []):
            qa_text += f"Q: {qa.get('question', '')}\n   Candidate Answer: {qa.get('candidate_answer', '')}\n   Correct Answer: {qa.get('correct_answer', '')}\n"
        
        user_message = f"""Please evaluate the candidate's listening comprehension.

        Passage:
        {kwargs.get('passage', '')}

        Questions and Candidate Answers:
        {qa_text}

        Use the evaluate_listening tool to assess comprehension."""
    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")
    
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": user_message
            }
        ]
    })

    structured: EvaluationOutput = result.get("structured_response") or result
    print(f"EVALUATION_AGENT: Structured output received: {structured}")

    return {
        "message": f"{evaluation_type} evaluation completed",
        "evaluation_type": evaluation_type,
        "evaluation": structured.model_dump() if hasattr(structured, 'model_dump') else structured,
        "status": "success"
    }
