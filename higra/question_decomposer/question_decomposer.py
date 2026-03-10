import json
import asyncio
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic

from higra_agent.logger import logger
from higra_agent.model_cache import ModelCache
from higra_agent.llm_client.llm_client import LLMAsyncClient
from higra_agent.prompt.question_decomposer_prompt import QuestionsDecomposerPrompt



class QuestionDecomposer(LLMAsyncClient):
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        logger_instance = logger,  # Allow custom logger
    ) -> None:
        
        super().__init__(api_key, model_name, logger_instance)

    async def run(self, question: str):

        ### Analyze Linguistic Features
        question_linguistic_features = self.analyze_question_hint(question)
        question_linguistic_features = json.dumps(
            question_linguistic_features, 
            indent=4, 
            ensure_ascii=False
        )
        logger.info(f"[Question Linguistic Features]:\n{question_linguistic_features}")

        ### Get Prompt
        system_prompt = QuestionsDecomposerPrompt.system_prompt
        example_prompt = QuestionsDecomposerPrompt.example_prompt
        user_prompt = QuestionsDecomposerPrompt.user_prompt.format(
            question=question,
            question_linguistic_features=question_linguistic_features
        )

        ### Call LLM
        response, usage = await self.call_single(
                system_prompt,
                example_prompt+user_prompt,
                response_format={ "type": "json_object" }
            )
        
        ### Parse Json
        # response = self.parse_json(response)
        usage['action'] = "[decompose]"

        ### Log Result
        logger.info(f"[Question Decomposition]:\n{json.dumps(response, ensure_ascii=True, indent=4)}")
        
        return response, [usage]

    def analyze_question_hint(self, question: str) -> dict:
        full_rel = []
        reduced_rel = []
        of_structs = []
        poss_structs = []
    
        if type(question) is list:
            question = question[0]
            
        doc = ModelCache._spacy_model(question)
    
        # Full relative clauses (relcl)
        for token in doc:
            if token.dep_ == "relcl":
                clause_span = doc[token.left_edge.i : token.right_edge.i + 1]
                head = token.head
                # Try to find a noun chunk for the head to avoid overextension
                head_span = next((chunk for chunk in doc.noun_chunks if head.i in range(chunk.start, chunk.end)), None)
                head_text = head_span.text if head_span else head.text
                full_rel.append(head_text + " " + clause_span.text)
    
        # Reduced relative clauses (acl)
        for token in doc:
            if token.dep_ == "acl":
                clause_span = doc[token.left_edge.i : token.right_edge.i + 1]
                head = token.head
                head_span = next((chunk for chunk in doc.noun_chunks if head.i in range(chunk.start, chunk.end)), None)
                head_text = head_span.text if head_span else head.text
                reduced_rel.append(head_text + " " + clause_span.text)
    
        # "of" structures
        for token in doc:
            if token.text.lower() == "of" and token.dep_ == "prep":
                head = token.head
                head_span = next((chunk for chunk in doc.noun_chunks if head.i in range(chunk.start, chunk.end)), None)
                pobj = next((child for child in token.children if child.dep_ == "pobj"), None)
                if pobj:
                    object_span = doc[pobj.left_edge.i : pobj.right_edge.i + 1]
                    of_structs.append((head_span.text if head_span else head.text) + " of " + object_span.text)
    
        # Possessive 's structures
        for token in doc:
            if token.dep_ == "poss" and token.tag_ in {"NN", "NNP", "NNS", "NNPS"}:
                start = token.head.left_edge.i
                end = token.head.right_edge.i + 1
                poss_structs.append(doc[start:end].text)
    
        return {
            "full_relative_clauses": full_rel,
            "reduced_relative_clauses": reduced_rel,
            "of_structures": of_structs,
            "possessive_s_structures": poss_structs
        }