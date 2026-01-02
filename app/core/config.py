import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    
    app_name: str = Field(default="PollFlow LLM Service", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment: development, production")
    debug: bool = Field(default=True, description="Debug mode")
    
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, description="Port to bind to")
    workers: int = Field(default=1, description="Number of worker processes")
    
    ollama_url: str = Field(
        default="http://localhost:11434/api/generate",
        description="Ollama API endpoint"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama base URL for health checks"
    )
    llm_model: str = Field(default="mixtral:8x7b", description="LLM model name", validation_alias="MODEL_NAME")
    
    default_temperature: float = Field(default=0.1, description="LLM temperature (0.0-1.0)")
    default_top_p: float = Field(default=0.9, description="LLM top_p value")
    default_num_predict: int = Field(default=512, description="Maximum tokens to generate")
    request_timeout: int = Field(default=300, description="LLM request timeout in seconds")
    
    min_questions: int = Field(default=1, description="Minimum number of questions")
    max_questions: int = Field(default=10, description="Maximum number of questions")
    min_answers: int = Field(default=2, description="Minimum answers per question")
    max_answers: int = Field(default=5, description="Maximum answers per question")
    
    cors_origins: str = Field(default="*", description="Comma-separated CORS origins")
    cors_methods: str = Field(default="*", description="Comma-separated HTTP methods")
    cors_headers: str = Field(default="*", description="Comma-separated headers")
    
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def cors_origins_list(self) -> list:
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"
    
    def get_ollama_health_url(self) -> str:
        return f"{self.ollama_base_url}/api/tags"


settings = Settings()

