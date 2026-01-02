import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "2.0.0"


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model" in data
    assert "ollama_status" in data
    assert "version" in data


def test_generate_survey_valid():
    payload = {
        "theme": "Employee Satisfaction",
        "questionCount": 2,
        "answersPerQuestion": 3
    }
    response = client.post("/generate-survey", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "questions" in data or "errorMessage" in data


def test_generate_survey_invalid_theme():
    payload = {
        "theme": "AB",
        "questionCount": 3,
        "answersPerQuestion": 4
    }
    response = client.post("/generate-survey", json=payload)
    assert response.status_code == 422


def test_generate_survey_invalid_question_count():
    payload = {
        "theme": "Valid Theme",
        "questionCount": 20,
        "answersPerQuestion": 4
    }
    response = client.post("/generate-survey", json=payload)
    assert response.status_code == 400


def test_models_endpoint():
    response = client.get("/models")
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "current_model" in data

