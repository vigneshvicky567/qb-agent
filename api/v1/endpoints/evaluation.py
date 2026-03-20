from fastapi import APIRouter, File, UploadFile, Form
from typing import Optional, List, Dict, Any
import json
from agents.evaluation_agent import run_evaluation_agent
from services.audio_service import AudioService

router = APIRouter()
audio_service = AudioService()

@router.post("/")
async def run_evaluation(
    evaluation_type: str = Form("grammar"), 
    candidate_id: str = Form("default_id"),
    original_passage: str = Form(""),
    transcribed_text: str = Form(""),
    audio_duration_seconds: int = Form(0),
    passage: str = Form(""),
    questions_and_answers: str = Form("[]"),
    audio_file: Optional[UploadFile] = File(None)
):
    try:
        qa_parsed = json.loads(questions_and_answers)
    except Exception:
        qa_parsed = []

    if audio_file and audio_file.filename:
        transcript_data = await audio_service.transcribe_audio(audio_file)
        transcribed_text = transcript_data.get("transcript", transcribed_text)

    result = await run_evaluation_agent(
        evaluation_type=evaluation_type,
        original_passage=original_passage,
        transcribed_text=transcribed_text,
        audio_duration_seconds=audio_duration_seconds,
        passage=passage,
        questions_and_answers=qa_parsed
    )
    return {"status": "success", "evaluation": result}


