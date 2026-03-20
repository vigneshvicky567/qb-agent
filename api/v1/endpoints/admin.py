from fastapi import APIRouter
from pydantic import BaseModel, Field
from agents.admin_agent import run_agent

router = APIRouter()

class QuestionBankRequest(BaseModel):
    count: int = Field(default=5, ge=1, description="Number of questions to generate")
    topic: str = Field(default="General English", description="Topic for the questions")
    difficulty: str = Field(default="Intermediate", description="Difficulty level")
    type: str = Field(default="grammar", description="Type of questions")

@router.post("/qb")
async def administer_question_bank(req: QuestionBankRequest):
    result = await run_agent(type=req.type, topic=req.topic, difficulty=req.difficulty, count=str(req.count))
    return {"status": "success", "result": result}


