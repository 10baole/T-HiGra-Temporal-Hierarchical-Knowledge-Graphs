from enum import Enum
from typing import Optional, Dict, Any
from higra_agent.llm_client.llm_client import LLMAsyncClient
from higra_agent.logger import logger

class QueryIntent(str, Enum):
    EXPLICIT_TEMPORAL = "explicit_temporal" # "In 2010..."
    IMPLICIT_TEMPORAL = "implicit_temporal" # "During Obama's presidency..."
    ORDERING = "ordering"                   # "Who came first..."
    ATEMPORAL = "atemporal"                 # "What is a bishop?"

class QueryIntentClassifier:
    def __init__(self, llm_client: LLMAsyncClient):
        self.llm_client = llm_client

    async def classify(self, question: str) -> QueryIntent:
        prompt = f"""
Analyze the temporal intent of this question: "{question}"

Classify into one of these categories:
1. explicit_temporal: Asks about a specific time (e.g., "In 2010", "on Jan 1st").
2. implicit_temporal: Asks about a time period defined by an event/entity (e.g., "During WW2", "When Obama was president").
3. ordering: Asks for temporal comparison (e.g., "Who was first?", "Before X...").
4. atemporal: Asks for definitions, facts that are generally true, or timeless concepts.

Return JSON:
{{
    "intent": "category_name"
}}
"""
        try:
            response, usage = await self.llm_client.call_single(
                system_prompt="You are a query intent classifier.",
                user_prompt=prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "intent_classification",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "intent": {
                                    "type": "string",
                                    "enum": ["explicit_temporal", "implicit_temporal", "ordering", "atemporal"]
                                }
                            },
                            "required": ["intent"],
                            "additionalProperties": False
                        }
                    }
                },
                max_tokens=50
            )
            # Response is a list (if using call_multiple) or single item? call_single returns (result, usage)
            # If response_format is used, result might be parsed object or dict.
            # Based on LLMClient implementation:
            # result = parsed if parsed is not None else self.parse_json(response.choices[0].message.content)
            
            # Since we pass a raw dict as response_format (not pydantic), it might return parsed JSON dict if strict=True works with beta.parse?
            # Or it might return string if beta.parse not triggered correctly with dict?
            # Let's assume it returns a dict.
            
            if isinstance(response, str):
                import json
                response = json.loads(response)
                
            intent_str = response.get("intent") #type: ignore
            return QueryIntent(intent_str)
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}. Defaulting to ATEMPORAL.")
            return QueryIntent.ATEMPORAL
