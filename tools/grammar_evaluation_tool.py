from pydantic import BaseModel
from typing import List, Optional
from langchain_openai import ChatOpenAI
import os 
from dotenv import load_dotenv
from difflib import SequenceMatcher

load_dotenv()

class GrammarEvaluationResult(BaseModel):
    accuracy_score: float  # 0-100
    error_count: int
    error_types: List[str]
    feedback: str
    overall_assessment: str

class GrammarEvaluationOutput(BaseModel):
    evaluation: GrammarEvaluationResult

def evaluate_grammar(original_passage: str, transcribed_text: str) -> str:
    """
    Evaluate grammar by comparing the transcribed text with the original passage.
    Checks for accuracy in transcription and identifies grammar/pronunciation errors.
    
    Args:
        original_passage (str): The original passage that should be read.
        transcribed_text (str): The candidate's transcribed/spoken text.
    
    Returns:
        str: JSON string with grammar evaluation results.
    """
    print(f"TOOL: Evaluating grammar - comparing original vs transcribed text")
    
    max_tokens = int(os.getenv("MAX_TOKENS_FROM_TOOLS", 1500))
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=max_tokens, temperature=0.3)
    structured_llm = llm.with_structured_output(GrammarEvaluationOutput, method="function_calling")
    
    # Calculate basic similarity
    similarity = SequenceMatcher(None, original_passage.lower(), transcribed_text.lower()).ratio()
    accuracy_score = round(similarity * 100, 2)
    
    prompt = f"""You are an English grammar and speech evaluation expert.

      Evaluate the candidate's transcribed text against the original passage for grammar accuracy, pronunciation patterns, and fluency.

      ORIGINAL PASSAGE:
      {original_passage}

      CANDIDATE'S TRANSCRIBED TEXT:
      {transcribed_text}

      Provide evaluation in JSON format with:
      1. accuracy_score (0-100): How closely the transcription matches the original
      2. error_count: Number of significant errors
      3. error_types: List of error categories (e.g., "word omission", "word substitution", "grammar error", "pronunciation error")
      4. feedback: Specific corrections needed
      5. overall_assessment: Brief overall assessment

      Base the accuracy_score on the similarity ratio of approximately {accuracy_score}% and analyze the quality of differences."""

    result: GrammarEvaluationOutput = structured_llm.invoke(prompt)
    return result.model_dump_json()
