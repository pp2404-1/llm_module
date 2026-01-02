import requests
import logging
from typing import Dict, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    
    def __init__(self):
        self.ollama_url = settings.ollama_url
        self.model_name = settings.llm_model
        self.timeout = settings.request_timeout
        
    def generate(self, prompt: str, options: Optional[Dict] = None) -> str:
        logger.info(f"Sending request to Ollama with model {self.model_name}")
        
        if options is None:
            options = {
                "temperature": settings.default_temperature,
                "top_p": settings.default_top_p,
                "num_predict": settings.default_num_predict
            }
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": options
        }
        
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            generated_text = result.get('response', '')
            
            logger.info(f"Successfully received response from Ollama ({len(generated_text)} chars)")
            return generated_text
            
        except requests.exceptions.Timeout:
            error_msg = f"LLM service timeout - модель не ответила за {self.timeout} секунд"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except requests.exceptions.ConnectionError as e:
            error_msg = "Не удалось подключиться к Ollama. Проверьте, запущен ли Ollama"
            logger.error(f"{error_msg}: {str(e)}")
            raise Exception(error_msg)
            
        except Exception as e:
            error_msg = f"Ошибка при вызове LLM: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def check_health(self) -> Dict:
        try:
            health_url = settings.get_ollama_health_url()
            logger.debug(f"Checking Ollama health at {health_url}")
            
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_available = any(
                    m.get('name') == self.model_name for m in models
                )
                
                if model_available:
                    logger.info(f"Ollama is healthy, model {self.model_name} is available")
                    return {
                        "status": "healthy",
                        "model_available": True,
                        "message": f"Model {self.model_name} is available"
                    }
                else:
                    logger.warning(f"Ollama is running but model {self.model_name} not found")
                    available_models = [m.get('name') for m in models]
                    return {
                        "status": "partial",
                        "model_available": False,
                        "message": f"Model {self.model_name} not found. Available models: {available_models}"
                    }
            else:
                logger.error(f"Ollama health check failed with status {response.status_code}")
                return {
                    "status": "unhealthy",
                    "model_available": False,
                    "message": f"Ollama returned status {response.status_code}"
                }
                
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama service")
            return {
                "status": "unhealthy",
                "model_available": False,
                "message": "Cannot connect to Ollama. Is it running?"
            }
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {
                "status": "unhealthy",
                "model_available": False,
                "message": f"Error: {str(e)}"
            }

