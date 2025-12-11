"""
Main application entry point
FastAPI application for LLM-powered survey generation
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import router

# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler
    Runs on startup and shutdown
    """
    # Startup
    logger.info("=" * 80)
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"📍 Environment: {settings.environment}")
    logger.info(f"🤖 Model: {settings.llm_model}")
    logger.info(f"🔗 Ollama URL: {settings.ollama_url}")
    logger.info(f"🌐 Server: {settings.host}:{settings.port}")
    logger.info(f"📊 Debug Mode: {settings.debug}")
    logger.info("=" * 80)
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Service...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    🤖 AI-powered survey generation service using Mixtral 8x7B via Ollama
    
    ## Features
    
    * **AI Survey Generation**: Generate contextual survey questions based on themes
    * **Flexible Configuration**: Customize question and answer counts
    * **Health Monitoring**: Check service and Ollama health status
    * **Production Ready**: Structured codebase with proper error handling
    
    ## Integration
    
    This service is designed to integrate with the PollFlow backend (Spring Boot) 
    and frontend (React) applications.
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    debug=settings.debug
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )

