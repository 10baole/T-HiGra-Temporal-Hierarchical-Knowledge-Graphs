import os
import re
import json
import yaml
import asyncio
from tqdm import tqdm
from openai import AsyncOpenAI
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic

from higra_agent.logger import logger  # Assuming this is a module-level logger; could be passed in.



T = TypeVar('T')  # Generic type for structured responses


class LLMAsyncClient(Generic[T]):
    """
    Async OpenAI client supporting structured and unstructured responses.
    Refactored for better modularity, typing, and error handling.
    """

    DEFAULT_HYPERPARAMS = {
        "temperature": 0.0,
        "max_tokens": 10000,  # Lowered default to a safer value for most models
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "seed": 42,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        logger_instance = logger,  # Allow custom logger
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_KEY")
        if not api_key:
            raise ValueError("API key must be provided or set in OPENAI_KEY env var.")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME") or "gpt-4o"  # Added fallback model
        if not self.model_name:
            raise ValueError("Model name must be provided or set in OPENAI_MODEL_NAME env var.")
        self.logger = logger_instance

    async def call_single(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Any] = None,  # e.g., Pydantic model or JSON schema
        retries: int = 0,  # Added retry support
        **hyperparams: Any,
    ) -> tuple[Union[str, T, Dict, List, None], Dict[str, int]]:
        """
        Makes a single API call. Supports structured outputs if response_format is provided.
        Retries on failure up to 'retries' times.
        """
        params = {**self.DEFAULT_HYPERPARAMS, **hyperparams}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        attempt = 0
        while attempt <= retries:
            try:
                if response_format:
                    response = await self.client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=messages,
                        response_format=response_format,
                        **params,
                    )
                    parsed = response.choices[0].message.parsed
                    result = parsed if parsed is not None else self.parse_json(response.choices[0].message.content)
                else:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **params,
                    )
                    result = response.choices[0].message.content

                return result, self._extract_usage(response)

            except Exception as e:
                attempt += 1
                self.logger.error(f"API call failed (attempt {attempt}/{retries + 1}): {str(e)}")
                if attempt > retries:
                    self.logger.error(f"Max retries exceeded. Returning None.")
                    return None, {}
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        return (None if response_format else ""), {}  # Fallback, though raise should prevent this

    async def call_multiple(
        self,
        message_list: List[tuple[str, str]],
        response_format: Optional[Any] = None,
        retries: int = 0,
        **hyperparams: Any,
    ) -> List[tuple[Union[str, T, Dict, List, None], Dict[str, int]]]:
        """
        Runs multiple API calls asynchronously. Aggregates results and logs errors individually.
        """
        tasks = [
            self.call_single(system_prompt, user_prompt, response_format, retries, **hyperparams)
            for system_prompt, user_prompt in message_list
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)  # Return exceptions for caller to handle

    async def call_multiple_batched(
        self,
        data: List[tuple[str, str]],
        batch_size: int = 200,
        response_format: Optional[Any] = None,
        retries: int = 0,
        **hyperparams: Any,
    ) -> List[tuple[Union[str, T, Dict, List, None], Dict[str, int]]]:
        """
        Processes data in batches to avoid overwhelming the API.
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
    
        results = []
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            batch_results = await self.call_multiple(batch, response_format, retries, **hyperparams)
            results.extend(batch_results)
    
        return results

    @staticmethod
    def _extract_usage(response) -> Dict[str, int]:
        """
        Extracts usage statistics from the OpenAI response.
        """
        usage = dict(response.usage or {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    @classmethod
    def _remove_code_fence(cls, text: str) -> str:
        # Remove leading/trailing code fences
        text = re.sub(r"^```(?:json|yaml)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # Fix invalid escape sequences (e.g., \' → ')
        text = re.sub(r'\\([^"\\/bfnrtu])', r'\1', text)

        return text

    @classmethod
    def parse_json(cls, json_string: str) -> Optional[Dict[str, Any]]:
        try:
            if json_string is None:
                logger.error(f"Error extracting JSON: received None (likely rate limited)")
                return None
                
            json_string = cls._remove_code_fence(json_string)
            return json.loads(json_string)
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}\n{json_string}")
        return None

    @classmethod
    def parse_yaml(cls, yaml_string: str) -> Optional[Dict[str, Any]]:
        try:
            yaml_string = cls._remove_code_fence(yaml_string)
            return yaml.safe_load(yaml_string)
        except Exception as e:
            logger.error(f"Error parsing YAML: {e}\n{yaml_string}")
        return None