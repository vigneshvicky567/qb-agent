from pydantic import BaseModel
from typing import Dict, Literal, List
from langchain_openai import ChatOpenAI
from agents.prompts import COMPREHENSION_PASSAGE_TOOL_PROMPT_TEMPLATE
import os
from dotenv import load_dotenv

load_dotenv()

# pydantic models to define the structure of the comprehension passage and questions
class ComprehensionQuestion(BaseModel):
    topic: str
    difficulty: str
    passage: str
    question: str
    options: Dict[Literal["a", "b", "c", "d"], str]
    correct_answer: Literal["a", "b", "c", "d"]
    explanation: str

class ComprehensionPassage(BaseModel):
    questions: List[ComprehensionQuestion]


def generate_comprehension_passages(topic: str, difficulty: str, count: int = 3) -> str:
    """
    Generate one reading comprehension passage followed by 'count' MCQ questions based on the passage.
    
    Args:
        topic (str): The topic for which to generate the comprehension passage.
        difficulty (str): The difficulty level (e.g., "beginner", "intermediate", "advanced").
        count (int): The number of MCQ questions to generate based on the passage. Default is 3.
    """

    print(f"TOOL: Generating {count} {difficulty} comprehension passages for topic: {topic}")

    # let's connect to LLM and generate questions based on the topic and difficulty
    max_tokens = int(os.getenv("MAX_TOKENS_FROM_TOOLS", 1500))
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=max_tokens, temperature=0.5)
    # telling llm to produce output in a structured format that matches our Passage model
    structured_llm = llm.with_structured_output(ComprehensionPassage, method="function_calling")

    # Let's invoke with a prompt
    prompt = (
        COMPREHENSION_PASSAGE_TOOL_PROMPT_TEMPLATE
        .replace("{topic}", topic)
        .replace("{difficulty}", difficulty)
        .replace("{count}", str(count))
    )
    result: ComprehensionPassage = structured_llm.invoke(prompt)

    return result.model_dump_json()