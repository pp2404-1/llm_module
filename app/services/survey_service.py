"""
Survey generation service
Business logic for generating surveys using LLM
"""

import json
import logging
import time
from typing import List, Dict
from app.models.schemas import SurveyRequest, QuestionItem, SurveyResponse
from app.services.ollama_service import OllamaService
from app.core.config import settings

logger = logging.getLogger(__name__)


class SurveyService:
    """Service for generating surveys using AI"""
    
    def __init__(self):
        self.ollama_service = OllamaService()
    
    def generate_survey(self, request: SurveyRequest) -> SurveyResponse:
        """
        Generate a survey based on the request
        
        Args:
            request: Survey generation request
            
        Returns:
            Generated survey response
        """
        start_time = time.time()
        
        try:
            logger.info(f"📝 Generating survey for theme: '{request.theme}'")
            logger.info(f"   Questions: {request.questionCount}, Answers per question: {request.answersPerQuestion}")
            
            # Create prompt for LLM
            prompt = self._create_prompt(request)
            logger.debug(f"Created prompt ({len(prompt)} chars)")
            
            # Call LLM
            llm_response = self.ollama_service.generate(prompt)
            logger.debug(f"Received LLM response ({len(llm_response)} chars)")
            
            # Parse and validate response
            questions = self._parse_and_validate(llm_response, request)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Successfully generated {len(questions)} questions in {processing_time:.2f}s")
            
            return SurveyResponse(
                questions=questions,
                processing_time=processing_time,
                model=settings.llm_model,
                errorMessage=None
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"❌ Error generating survey: {error_msg}")
            
            return SurveyResponse(
                questions=None,
                processing_time=processing_time,
                model=settings.llm_model,
                errorMessage=error_msg
            )
    
    def _create_prompt(self, request: SurveyRequest) -> str:
        """
        Create prompt for LLM based on request
        
        Args:
            request: Survey generation request
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Ты есть REST API backend. Ты отвечаешь в валидном JSON формате.
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
            "answers": ["ответ 1", "ответ 2", "ответ 3", "ответ 4"]
        }}
    ]
}}

Не добавляй никаких пояснений, комментариев или форматирования кроме JSON.
Генерируй именно {request.questionCount} вопросов с {request.answersPerQuestion} ответами каждый.
"""
        return prompt.strip()
    
    def _parse_and_validate(
        self, 
        llm_response: str, 
        request: SurveyRequest
    ) -> List[QuestionItem]:
        """
        Parse and validate LLM response
        
        Args:
            llm_response: Raw response from LLM
            request: Original request for validation
            
        Returns:
            List of validated questions
            
        Raises:
            ValueError: If response is invalid
        """
        try:
            # Extract JSON from response
            json_data = self._extract_json(llm_response)
            
            # Validate structure
            if "questions" not in json_data:
                raise ValueError("Неверная структура ответа: отсутствует поле 'questions'")
            
            questions_data = json_data["questions"]
            if not isinstance(questions_data, list):
                raise ValueError("Поле 'questions' должно быть массивом")
            
            if len(questions_data) == 0:
                raise ValueError("LLM не сгенерировал ни одного вопроса")
            
            # Parse and validate each question
            questions = []
            for i, q_data in enumerate(questions_data):
                if not isinstance(q_data, dict):
                    logger.warning(f"⚠️ Question {i+1} is not a dictionary, skipping")
                    continue
                
                if "question" not in q_data or "answers" not in q_data:
                    logger.warning(f"⚠️ Question {i+1} missing required fields, skipping")
                    continue
                
                if not isinstance(q_data["answers"], list):
                    logger.warning(f"⚠️ Question {i+1} answers is not a list, skipping")
                    continue
                
                # Clean and validate
                question_text = str(q_data["question"]).strip()
                answers = [str(ans).strip() for ans in q_data["answers"] if str(ans).strip()]
                
                # Ensure correct number of answers
                if len(answers) > request.answersPerQuestion:
                    answers = answers[:request.answersPerQuestion]
                elif len(answers) < request.answersPerQuestion:
                    logger.warning(
                        f"⚠️ Question {i+1} has only {len(answers)} answers, "
                        f"expected {request.answersPerQuestion}"
                    )
                
                if len(answers) >= 2:  # Minimum 2 answers required
                    try:
                        question_item = QuestionItem(
                            question=question_text,
                            answers=answers
                        )
                        questions.append(question_item)
                    except Exception as e:
                        logger.warning(f"⚠️ Question {i+1} validation failed: {e}")
            
            if len(questions) == 0:
                raise ValueError("Не удалось извлечь валидные вопросы из ответа LLM")
            
            if len(questions) < request.questionCount:
                logger.warning(
                    f"⚠️ Generated {len(questions)} questions, "
                    f"requested {request.questionCount}"
                )
            
            return questions
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parse error: {e}")
            logger.error(f"Raw response: {llm_response[:500]}")
            raise ValueError(f"Невалидный JSON от LLM: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            raise
    
    def _extract_json(self, text: str) -> Dict:
        """
        Extract JSON object from text
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If no valid JSON found
        """
        # Try to find JSON object in the text
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("JSON не найден в ответе LLM")
        
        json_str = text[start_idx:end_idx]
        logger.debug(f"Extracted JSON ({len(json_str)} chars): {json_str[:200]}...")
        
        return json.loads(json_str)

