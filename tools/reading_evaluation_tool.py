from pydantic import BaseModel
from typing import Optional, Literal
from langchain_openai import ChatOpenAI
import os 
from dotenv import load_dotenv

load_dotenv()

class ReadingEvaluationResult(BaseModel):
    reading_speed_wpm: float
    reading_speed_assessment: Literal["below_average", "average", "above_average", "excellent"]
    fluency_score: float  # 0-100
    pronunciation_accuracy: float  # 0-100
    overall_score: float  # 0-100
    feedback: str
    recommendations: str

class ReadingEvaluationOutput(BaseModel):
    evaluation: ReadingEvaluationResult

def evaluate_reading(original_passage: str, transcribed_text: str, audio_duration_seconds: float, min_wpm: int = 140, max_wpm: int = 170) -> str:
    """
    Evaluate reading skills based on speed, fluency, and pronunciation accuracy.
    
    Args:
        original_passage (str): The original passage to be read.
        transcribed_text (str): The candidate's transcribed speech.
        audio_duration_seconds (float): Duration of the audio in seconds.
        min_wpm (int): Minimum acceptable words per minute (default 140).
        max_wpm (int): Maximum acceptable words per minute (default 170).
    
    Returns:
        str: JSON string with reading evaluation results.
    """
    print(f"TOOL: Evaluating reading - speed, fluency, and pronunciation")
    
    # Calculate reading speed
    transcribed_words = len(transcribed_text.split())
    audio_duration_minutes = audio_duration_seconds / 60
    reading_speed_wpm = round(transcribed_words / audio_duration_minutes, 2) if audio_duration_minutes > 0 else 0
    
    # Determine reading speed assessment
    if reading_speed_wpm < min_wpm:
        speed_assessment = "below_average"
    elif reading_speed_wpm > max_wpm:
        speed_assessment = "above_average"
    else:
        speed_assessment = "average"
    
    max_tokens = int(os.getenv("MAX_TOKENS_FROM_TOOLS", 1500))
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=max_tokens, temperature=0.3)
    structured_llm = llm.with_structured_output(ReadingEvaluationOutput, method="function_calling")
    
    prompt = f"""You are an expert English reading and pronunciation evaluator.

Evaluate the candidate's reading performance based on the provided metrics and transcription quality.

ORIGINAL PASSAGE:
{original_passage}

CANDIDATE'S TRANSCRIBED TEXT:
{transcribed_text}

READING METRICS:
- Words Per Minute (WPM): {reading_speed_wpm}
- Expected Range: {min_wpm}-{max_wpm} WPM
- Reading Speed Assessment: {speed_assessment}
- Total words spoken: {transcribed_words}
- Audio duration: {audio_duration_seconds} seconds

Provide evaluation in JSON format with:
1. reading_speed_wpm: {reading_speed_wpm} (calculated value)
2. reading_speed_assessment: "{speed_assessment}" (categorize accordingly)
3. fluency_score (0-100): Based on smooth delivery and pauses
4. pronunciation_accuracy (0-100): Based on transcription accuracy compared to original
5. overall_score (0-100): Weighted average of all metrics
6. feedback: Specific observations about the reading performance
7. recommendations: Actionable improvement suggestions

Consider natural speaking patterns and transcription accuracy in your evaluation."""

    result: ReadingEvaluationOutput = structured_llm.invoke(prompt)
    return result.model_dump_json()
