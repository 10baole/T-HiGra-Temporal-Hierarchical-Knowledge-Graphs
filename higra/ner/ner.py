import json
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic

from higra_agent.logger import logger
from higra_agent.llm_client.llm_client import LLMAsyncClient
from higra_agent.prompt.ner_prompt import NERPrompt



class NameEntityRecognizer(LLMAsyncClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        logger_instance = logger,  # Allow custom logger
    ) -> None:
        
        super().__init__(api_key, model_name, logger_instance)
    
    async def run(self, query, top_related_entities):
        
        ### Get Prompt
        system_prompt = NERPrompt.system_prompt
        example_prompt = NERPrompt.example_prompt
        user_prompt = NERPrompt.user_prompt.format(
            query=query,
            top_related_entities=set(top_related_entities)
        )

        ### Call LLM
        response, usage = await self.call_single(
                system_prompt,
                example_prompt+user_prompt,
                # response_format={ "type": "json_object" }
            )
        
        ### Parse Json
        result, usage = self.parse_json((response, usage))
        logger.info(f"[Named Entity Recognizer]:\n{json.dumps(result, ensure_ascii=True, indent=4)}")
        return result, [usage]

    async def run_multiple(self, query_list, top_related_entities_list):

        for top_entities in top_related_entities_list:
            logger.info(f"[Named Entity Recognizer]:Top Related Entities\n{json.dumps(top_entities, ensure_ascii=True, indent=4)}")
            
        ### Get Prompt
        prompt_list = [
            (
                NERPrompt.system_prompt,
                NERPrompt.example_prompt+NERPrompt.user_prompt.format(
                    query=query,
                    top_related_entities=set(top_related_entities)
                )
            )
            for query, top_related_entities in zip(query_list, top_related_entities_list)
        ]

        ### Call LLM
        result_list = await self.call_multiple_batched(
            prompt_list,
            # response_format={ "type": "json_object" }
        )

        ### Process Result
        result_list = [self.parse_json(x) for x in result_list]
        
        result = [x[0] for x in result_list]
        usage = [x[1] for x in result_list]

        for r in result:
            logger.info(f"[Named Entity Recognizer]:\n{json.dumps(r, ensure_ascii=True, indent=4)}")
        
        return result, usage

    def parse_json(self, llm_response):
        response, usage = llm_response
        try:
            result = super().parse_json(response)
            result = {
                # 'reasoning': result['reasoning'],
                # "entity_name": [x.replace("$", "") for x in result["entity_appeared_in_query"]] + [x.replace("$", "") for x in result["entity_similar_to_entity_appeared_in_query"]] 
                "entity_name": [x.replace("$", "") for x in result["entity_appeared_in_query"]]
            }
            if any([type(x) is not str for x in result['entity_name']]):
                result = {
                    'entity_name': []
                }
            usage['action'] = "[ner]"
        except Exception as e:
            logger.error(f"[Named Entity Recognizer] error: {e}\n{response}")
            result = {
                'reasoning': None,
                'entity_name': []
            }
            usage = {}

        return result, usage