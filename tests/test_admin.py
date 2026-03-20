from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_admin_qb():
    response = client.post("/api/v1/admin/qb", json={"count": 5})
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["status"] == "success"
    # Note: Will require OPENAI_API_KEY to test actual agent logic
    # assert "questions" in json_resp["result"]

