from pydantic import BaseModel
from typing import Dict, Literal, List, Optional
from langchain_openai import ChatOpenAI
from agents.prompts import GRAMMAR_MCQS_TOOL_PROMPT_TEMPLATE
import os 
from dotenv import load_dotenv

load_dotenv()

class GrammarMCQ(BaseModel):
    topic: str
    difficulty: str
    passage: Optional[str] = None
    question: str
    options: Dict[Literal["a", "b", "c", "d"], str]
    correct_answer: Literal["a", "b", "c", "d"]
    explanation: str

class GrammarMCQListOutput(BaseModel):
    questions: List[GrammarMCQ]

def generate_grammar_mcqs(topic: str, difficulty: str, count: int = 5) -> str:
    """
    Generate grammar MCQs based on the given topic and difficulty level. 
    You should give 4 options for each question and also specify the correct answer. 
    
    Args:
        topic (str): The grammar topic for which to generate questions.
        difficulty (str): The difficulty level of the questions (e.g., "beginner", "intermediate", "advanced").
        count (int): The number of questions to generate. Default is 5.
    """
    print(f"TOOL: Generating {count} {difficulty} MCQs for topic: {topic}")

    # let's connect to LLM and generate questions based on the topic and difficulty
    max_tokens = int(os.getenv("MAX_TOKENS_FROM_TOOLS", 1500))
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=max_tokens, temperature=0.5)
    # telling llm to produce output in a structured format that matches our GrammarMCQListOuput model
    structured_llm = llm.with_structured_output(GrammarMCQListOutput, method="function_calling")

    # Let's invoke with a prompt
    prompt = (
        GRAMMAR_MCQS_TOOL_PROMPT_TEMPLATE
        .replace("{topic}", topic)
        .replace("{difficulty}", difficulty)
        .replace("{count}", str(count))
    )
    result: GrammarMCQListOutput = structured_llm.invoke(prompt)

    # print(f"TOOL: Generated questions: {result}")
    return result.model_dump_json()