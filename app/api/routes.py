"""
API Routes
Defines all HTTP endpoints for the LLM service
"""

import logging
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import (
    SurveyRequest,
    SurveyResponse,
    HealthResponse,
    ErrorResponse
)
from app.services.survey_service import SurveyService
from app.services.ollama_service import OllamaService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
survey_service = SurveyService()
ollama_service = OllamaService()


@router.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - Service information
    """
    return {
        "message": "LLM Service is running!",
        "version": settings.app_version,
        "model": settings.llm_model,
        "environment": settings.environment,
        "docs": "/docs",
        "health": "/health"
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Check health of LLM service and Ollama connectivity"
)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthResponse: Service health status
    """
    try:
        # Check Ollama service
        ollama_health = ollama_service.check_health()
        ollama_status = ollama_health.get("status", "unknown")
        
        # Determine overall status
        service_status = "healthy" if ollama_status == "healthy" else "degraded"
        
        logger.info(f"Health check: service={service_status}, ollama={ollama_status}")
        
        return HealthResponse(
            status=service_status,
            model=settings.llm_model,
            ollama_status=ollama_status,
            version=settings.app_version,
            environment=settings.environment
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model=settings.llm_model,
            ollama_status=f"error: {str(e)}",
            version=settings.app_version,
            environment=settings.environment
        )


@router.post(
    "/generate-survey",
    response_model=SurveyResponse,
    tags=["Survey Generation"],
    summary="Generate Survey",
    description="Generate survey questions and answers using AI (Mixtral 8x7B)",
    responses={
        200: {
            "description": "Survey successfully generated",
            "model": SurveyResponse
        },
        400: {
            "description": "Invalid request",
            "model": ErrorResponse
        },
        500: {
            "description": "Server error",
            "model": ErrorResponse
        }
    }
)
async def generate_survey(request: SurveyRequest):
    """
    Generate a survey using AI based on a theme
    
    Args:
        request: Survey generation request with theme, question count, and answer count
        
    Returns:
        SurveyResponse: Generated survey with questions and answers
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        logger.info(f"📥 Received survey generation request: theme='{request.theme}'")
        
        # Validate request parameters
        if request.questionCount < settings.min_questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Question count must be at least {settings.min_questions}"
            )
        
        if request.questionCount > settings.max_questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Question count cannot exceed {settings.max_questions}"
            )
        
        if request.answersPerQuestion < settings.min_answers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Answers per question must be at least {settings.min_answers}"
            )
        
        if request.answersPerQuestion > settings.max_answers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Answers per question cannot exceed {settings.max_answers}"
            )
        
        # Generate survey
        response = survey_service.generate_survey(request)
        
        if response.errorMessage:
            logger.warning(f"⚠️ Survey generation completed with error: {response.errorMessage}")
        else:
            logger.info(f"✅ Survey generation successful: {len(response.questions or [])} questions")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in generate_survey endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/models",
    tags=["Models"],
    summary="List Available Models",
    description="Get list of available LLM models from Ollama"
)
async def list_models():
    """
    Get list of available models from Ollama
    
    Returns:
        Dict with available models
    """
    try:
        health = ollama_service.check_health()
        return {
            "current_model": settings.llm_model,
            "ollama_status": health.get("status"),
            "message": health.get("message")
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve models: {str(e)}"
        )

