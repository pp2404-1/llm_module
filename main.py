from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import json
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "mixtral:8x7b")

app = FastAPI(title="LLM Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class SurveyRequest(BaseModel):
    theme: str
    questionCount: int = 3
    answersPerQuestion: int = 2


class QuestionItem(BaseModel):
    question: str
    answers: List[str]


class SurveyResponse(BaseModel):
    questions: List[QuestionItem]
    processing_time: Optional[float] = None
    model: Optional[str] = None
    errorMessage: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    ollama_status: str


@app.get("/")
async def root():
    return {"message": "LLM Service is running!", "model": MODEL_NAME}


@app.get("/health")
async def health_check():
    """Check health of LLM service and Ollama"""
    try:
        # Check Ollama connection
        response = requests.get(f"http://localhost:11434/api/tags", timeout=10)
        ollama_status = "healthy" if response.status_code == 200 else "unhealthy"

        return HealthResponse(
            status="healthy",
            model=MODEL_NAME,
            ollama_status=ollama_status
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model=MODEL_NAME,
            ollama_status=f"error: {str(e)}"
        )


@app.post("/generate-survey", response_model=SurveyResponse)
async def generate_survey(request: SurveyRequest):
    start_time = time.time()

    try:
        logger.info(f"Generating survey for theme: {request.theme}")

        # Create prompt for LLM
        prompt = create_survey_prompt(request)
        logger.info(f"Created prompt with {len(prompt)} characters")

        # Call Ollama
        llm_response = call_ollama(prompt)
        logger.info(f"Received LLM response with {len(llm_response)} characters")

        # Parse JSON from LLM response
        parsed_data = parse_llm_response(llm_response)

        # Validate and transform data
        survey_response = validate_survey_data(parsed_data, request)
        survey_response.processing_time = time.time() - start_time
        survey_response.model = MODEL_NAME

        logger.info(f"Successfully generated survey in {survey_response.processing_time:.2f}s")

        return survey_response

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error generating survey: {str(e)}")

        return SurveyResponse(
            questions=[],
            processing_time=processing_time,
            model=MODEL_NAME,
            errorMessage=str(e)
        )


def create_survey_prompt(request: SurveyRequest) -> str:
    return f"""
    Сгенерируй {request.questionCount} вопросов для опроса сотрудников на тему '{request.theme}' 
    и {request.answersPerQuestion} возможных ответа на каждый вопрос.

    Требования:
    - Вопросы должны быть конкретными и относиться к теме
    - Ответы должны быть краткими и понятными
    - Используй русский язык
    - Верни ТОЛЬКО JSON без дополнительного текста

    Структура JSON должна быть точной:
    {{
        "questions": [
            {{
                "question": "текст вопроса",
                "answers": ["ответ 1", "ответ 2"]
            }}
        ]
    }}

    Не добавляй никаких пояснений, комментариев или форматирования кроме JSON.
    """


def call_ollama(prompt: str) -> str:
    """Call Ollama API with error handling"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 512
        }
    }

    try:
        logger.info(f"Sending request to Ollama with model {MODEL_NAME}")
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120.0  # 2 minutes timeout for LLM
        )

        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        result = response.json()
        return result.get('response', '')

    except requests.exceptions.Timeout:
        raise Exception("LLM service timeout - модель не ответила за 120 секунд")
    except requests.exceptions.ConnectionError:
        raise Exception("Не удалось подключиться к Ollama. Проверьте, запущен ли Ollama")
    except Exception as e:
        raise Exception(f"Ошибка при вызове LLM: {str(e)}")


def parse_llm_response(llm_response: str) -> dict:
    """Extract and parse JSON from LLM response"""
    try:
        # Clean the response and find JSON
        start_idx = llm_response.find('{')
        end_idx = llm_response.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("JSON не найден в ответе LLM")

        json_str = llm_response[start_idx:end_idx]
        logger.info(f"Extracted JSON: {json_str[:100]}...")

        return json.loads(json_str)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Raw LLM response: {llm_response}")
        raise ValueError(f"Невалидный JSON от LLM: {str(e)}")


def validate_survey_data(parsed_data: dict, request: SurveyRequest) -> SurveyResponse:
    """Validate data from LLM"""
    if "questions" not in parsed_data:
        raise ValueError("Неверная структура ответа: отсутствует поле 'questions'")

    questions = []
    for i, q_data in enumerate(parsed_data["questions"]):
        if not isinstance(q_data, dict):
            raise ValueError(f"Неверный формат вопроса в индексе {i}")

        if "question" not in q_data or "answers" not in q_data:
            raise ValueError(f"Отсутствуют обязательные поля в вопросе {i}")

        if not isinstance(q_data["answers"], list):
            raise ValueError(f"Answers should be a list in question {i}")

        # Ensure we have the correct number of answers
        answers = [str(ans).strip() for ans in q_data["answers"][:request.answersPerQuestion]]

        questions.append(QuestionItem(
            question=q_data["question"].strip(),
            answers=answers
        ))

    if len(questions) < request.questionCount:
        logger.warning(f"Requested {request.questionCount} questions, but got {len(questions)}")

    return SurveyResponse(questions=questions)


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting LLM Service...")
    logger.info(f"Ollama URL: {OLLAMA_URL}")
    logger.info(f"Model: {MODEL_NAME}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )