import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import router

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 80)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Model: {settings.llm_model}")
    logger.info(f"Ollama URL: {settings.ollama_url}")
    logger.info(f"Server: {settings.host}:{settings.port}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info("=" * 80)
    
    yield
    
    logger.info("Shutting down LLM Service...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered survey generation service using Mixtral 8x7B via Ollama",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    debug=settings.debug
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

