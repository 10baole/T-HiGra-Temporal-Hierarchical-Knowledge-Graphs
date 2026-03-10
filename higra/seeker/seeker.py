import re
import json
import asyncio
import aiohttp
import requests

from typing import Dict, Any, List, Optional, Union, TypeVar, Generic

from higra_agent.timer import Timer
from higra_agent.logger import logger
from higra_agent.librarian.librarian import Librarian
from higra_agent.seeker.action_schema import Action
from higra_agent.prompt.seeker_prompt import SeekerPrompt
from higra_agent.llm_client.llm_client import LLMAsyncClient
from higra_agent.question_clarifier.question_clarifier import QuestionClarifier
from higra_agent.question_decomposer.question_decomposer import QuestionDecomposer



class Seeker(LLMAsyncClient):
    
    def __init__(
        self,
        retriever,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        logger_instance = logger,  # Allow custom logger
        max_turns: int=5,
        retrieval_mode: str="higra_retriever" # Allow higra_retriever | hybrid
    ) -> None:
        super().__init__(api_key, model_name, logger_instance)


        self.librarian = Librarian(
            api_key=api_key,
            model_name=model_name,
            retriever=retriever
        )
        self.max_turns = max_turns

        self.question_clarifier = QuestionClarifier(
            api_key, 
            model_name,
            logger_instance
        )

        self.question_decomposer = QuestionDecomposer(
            api_key, 
            model_name,
            logger_instance
        )

        self.function_map = {
            "ask_librarian": self._ask_librarian,
            "answer": self._answer,
        }

        self.retrieval_mode=retrieval_mode

    def preprocess_question(self, question):
        question = question.replace("\"", "")
        return question
        
    async def run(
        self,
        question: str
    ):
        history = []
        usage = []
        timer = Timer()

        logger.info(f"[Question]: {question}")
        question =self.preprocess_question(question)
        async with timer.track("execution_time"):
            ### Rewrite
            question_start_word_pattern = re.compile(
                r"^(who|what|where|when|why|how|which)\b",
                re.IGNORECASE
            )

            question_word_pattern = re.compile(
                r"(who|what|where|when|why|how|which)\b",
                re.IGNORECASE
            )

            rewrite_question = question
            if not question_start_word_pattern.match(question) and question_word_pattern.search(question):
                rewirte_result, rewrite_usage =await self.question_clarifier.run(question)
                rewrite_question = rewirte_result['rewrite_question']
                usage+=rewrite_usage
            
            ### Decompose
            decompose_result, decompose_usage = await self.question_decomposer.run(rewrite_question)
            # decompose_result.pop('reasoning')
            usage+=decompose_usage
            # history+=[decompose_result]
                
            ### Async Triggers
            init_state, init_state_usage = await self._handling_decompose_result(
                decompose_result['question_decomposition']
            )
            usage+=init_state_usage
            
            ### Reasoning Loops
            history+=init_state
            is_done = False
            turn_count = 0
            
            system_prompt = SeekerPrompt.system_prompt
            example_prompt = SeekerPrompt.example_prompt
            
            while not is_done and turn_count < self.max_turns:
    
                ### Get Prompt
                user_prompt = SeekerPrompt.user_prompt.format(
                    history=history,
                    question=question,                
                    question_decomposition=decompose_result
                )
    
                ### Call LLM
                seeker_response, seeker_usage = await self.call_single(
                        system_prompt,
                        example_prompt+user_prompt,
                        response_format=Action,
                    )
                seeker_response = seeker_response.model_dump()
                seeker_usage['action'] = '[seeker]'
                usage+=[seeker_usage]
                
                logger.info(f"[Seeker]:\n{json.dumps(seeker_response, ensure_ascii=True, indent=4)}")
                
                ### Execute Action
                action_result, action_usage = await self._execute_action(
                        seeker_response['action']['function_name'],
                        {
                            param['param_name']: param['param_value'] 
                            for param in seeker_response['action']['function_parameters']
                        }
                    )
    
                ### Process Result
                turn_count+=1
                is_done = seeker_response['action']['function_name']=="answer"
                history+=[{
                    "turn": turn_count,
                    "action": seeker_response['action'],
                    "action_result": action_result,
                    
                }]
                usage+=action_usage
        
        return action_result, usage, timer.timings, history

    async def _handling_decompose_result(
        self,
        decompose_result
    ):
        init_questions = [
            x['subquestion']
            for x in decompose_result if x['depends_on']==None
        ]
        
        librarian_response, librarian_usage = await self._ask_librarian(init_questions)
        init_state = [
            {
                "turn": 0,
                "action": {
                    "function_name": "ask_librarian",
                    "function_parameters": [
                        {
                            "param_name": "question_list",
                            "param_value": init_questions
                        }
                    ]
                },
                "action_result": librarian_response,
                
            }
        ]
        
        return init_state, librarian_usage

    async def _ask_librarian(self, question_list):

        # Handle exception
        if type(question_list) is str:
            question_list = [question_list]

        question_list = [q for q in question_list]
        logger.info(f"[Seeker]: asking '{question_list}'")
        
        librarian_response, librarian_usage, librarian_context_list = await self.librarian.run_multiple(
            question_list,
            retrieval_mode=self.retrieval_mode
        )

        for response in librarian_response:
            if type(response) is dict and "reasoning" in response:
                response.pop("reasoning")
            
        return librarian_response, librarian_usage
    
    async def _answer(self, answer, evidence=None):
        final_result = {
            'answer': answer,
            'evidence': evidence
        }
        return final_result, []
    
    async def _execute_action(self, function_name, kwargs):
        parameter_list = list(kwargs.keys())
        parameter_list = [x for x in parameter_list if not (function_name == "answer" and x != "answer_string")]
        return await self.function_map[function_name](**kwargs)