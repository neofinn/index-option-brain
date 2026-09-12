from fastapi.testclient import TestClient

from index_option_brain.app.main import app


def test_health_endpoint_reports_llm_disabled_by_default():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_enabled"] is False
