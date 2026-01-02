from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SurveyRequest(BaseModel):
    
    theme: str = Field(
        ..., 
        min_length=3,
        max_length=200,
        description="Theme or topic for the survey",
        examples=["Удовлетворенность сотрудников"]
    )
    questionCount: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of questions to generate",
        examples=[3]
    )
    answersPerQuestion: int = Field(
        default=4,
        ge=2,
        le=5,
        description="Number of answer options per question",
        examples=[4]
    )
    
    @field_validator('theme')
    @classmethod
    def validate_theme(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Theme cannot be empty")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "theme": "Корпоративная культура",
                "questionCount": 3,
                "answersPerQuestion": 4
            }
        }


class QuestionItem(BaseModel):
    
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Question text"
    )
    answers: List[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="List of possible answers"
    )
    
    @field_validator('answers')
    @classmethod
    def validate_answers(cls, v: List[str]) -> List[str]:
        cleaned = [ans.strip() for ans in v if ans.strip()]
        if len(cleaned) < 2:
            raise ValueError("At least 2 non-empty answers required")
        return cleaned
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Как вы оцениваете корпоративную культуру?",
                "answers": ["Отлично", "Хорошо", "Удовлетворительно", "Плохо"]
            }
        }


class SurveyResponse(BaseModel):
    
    questions: Optional[List[QuestionItem]] = Field(
        default=None,
        description="Generated questions and answers"
    )
    processing_time: Optional[float] = Field(
        default=None,
        description="Processing time in seconds"
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model used"
    )
    errorMessage: Optional[str] = Field(
        default=None,
        description="Error message if generation failed"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "questions": [
                    {
                        "question": "Как вы оцениваете корпоративную культуру?",
                        "answers": ["Отлично", "Хорошо", "Удовлетворительно", "Плохо"]
                    }
                ],
                "processing_time": 2.5,
                "model": "mixtral:8x7b",
                "errorMessage": None
            }
        }


class HealthResponse(BaseModel):
    
    status: str = Field(..., description="Service status")
    model: str = Field(..., description="LLM model name")
    ollama_status: str = Field(..., description="Ollama service status")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Environment")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model": "mixtral:8x7b",
                "ollama_status": "healthy",
                "version": "2.0.0",
                "environment": "development"
            }
        }


class ErrorResponse(BaseModel):
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid request",
                "detail": "Theme must be at least 3 characters long"
            }
        }

