import json
import asyncio
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic

from higra_agent.logger import logger
from higra_agent.timer import Timer
from higra_agent.prompt.librarian_prompt import LibrarianPrompt
from higra_agent.llm_client.llm_client import LLMAsyncClient


class Librarian(LLMAsyncClient):
    
    def __init__(
        self,
        retriever,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        logger_instance = logger,  # Allow custom logger
    ) -> None:
        super().__init__(api_key, model_name, logger_instance)
        self.retriever = retriever

    async def run(
        self,
        question
    ):
        ### Retrieve Context
        context = await self.retriever.retrieve(
            question
        )

        additional_instruction = "(Return all possible, complete, and fully detailed answers of all possible interpretations, all timeline)"
        question=question+additional_instruction
        ### Get Prompt
        system_prompt = LibrarianPrompt.system_prompt
        example_prompt = LibrarianPrompt.example_prompt
        user_prompt = LibrarianPrompt.user_prompt.format(
            question=question,
            context=context
        )

        ### Call LLM
        response, usage = await self.call_single(
                system_prompt,
                example_prompt+user_prompt,
                response_format={ "type": "json_object" }
            )

        ### Parse Json
        response = self.parse_json(response)
        response['question'] = question
        usage['action'] = "[retrieve]"

        ### Log Result
        logger.info(f"[Librarian]:\n{json.dumps(response, ensure_ascii=True, indent=4)}")
        return response, [usage], context


    async def run_multiple(
        self,
        question_list,
        retrieval_mode="higra_retriever" # higra_retriever | hybrid
    ):

        if retrieval_mode=="higra_retriever":
        
            ### Retrieve Context
            context_list, context_usage = await self.retriever.retrieve_multiple(
                question_list
            )

        elif retrieval_mode=="hybrid":

            context_list, context_usage = await self.retriever.retrieve_multiple_hybrid(
                question_list
            )

        else:

            raise ValueError(
                "Invalid retrieval mode specified. Supported modes: 'higra_retriever', 'hybrid'."
            )


        additional_instruction = "(Return all possible, complete, and fully detailed answers of all possible interpretations, all timeline)"

        ### Get Prompt
        prompt_list = [
            (
                LibrarianPrompt.system_prompt,
                LibrarianPrompt.example_prompt + LibrarianPrompt.user_prompt.format(
                    question=question+additional_instruction,
                    context=context
                )
            )
            for question, context in zip(question_list, context_list)
        ]

        ### Call LLM
        response_list = await self.call_multiple_batched(
            prompt_list,
            response_format={ "type": "json_object" }
        )

        usage_list = [x[1] for x in response_list]
        response_list = [x[0] for x in response_list]

        for usage in usage_list:
            if type(usage) is dict:
                usage['action'] = "[librarian]"
        usage_list += context_usage
        
        ### Log Result
        for response, question in zip(response_list, question_list):
            response['question'] = question
            logger.info(f"[Librarian]:\n{json.dumps(response, ensure_ascii=True, indent=4)}")
        return response_list, usage_list, context_list