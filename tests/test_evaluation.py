from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_evaluation():
    response = client.post("/api/v1/evaluation/", data={"candidate_id": "c123", "passage": "Hello world", "evaluation_type": "reading", "audio_duration_seconds": 1})
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["status"] == "success"
    # Note: Requires OPENAI_API_KEY to hit the LLM and receive real agent format 
    # assert "evaluation" in json_resp

