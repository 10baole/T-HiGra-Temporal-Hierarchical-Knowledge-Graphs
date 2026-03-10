import json
import asyncio
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic

from higra_agent.logger import logger
from higra_agent.llm_client.llm_client import LLMAsyncClient
from higra_agent.prompt.question_clarifier_prompt import QuestionClarifierPrompt



class QuestionClarifier(LLMAsyncClient):
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        logger_instance = logger,  # Allow custom logger
    ) -> None:
        
        super().__init__(api_key, model_name, logger_instance)

    
    async def run(self, question):
        
        ### Get Prompt
        system_prompt = QuestionClarifierPrompt.system_prompt
        example_prompt = QuestionClarifierPrompt.example_prompt
        user_prompt = QuestionClarifierPrompt.user_prompt.format(
            question=question,
        )

        ### Call LLM
        response, usage = await self.call_single(
            system_prompt,
            example_prompt+user_prompt
        )

        ### Parse Json
        response = self.parse_json(response)
        usage['action'] = "[rewrite]"

        ### Log Result
        logger.info(f"[Question Rewrite]:\n{json.dumps(response, ensure_ascii=True, indent=4)}")
        
        return response, [usage]