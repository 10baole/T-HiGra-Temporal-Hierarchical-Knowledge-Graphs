import os
import json
import asyncio

from higra_agent.logger import logger
from higra_agent.model_cache import ModelCache

from higra_agent.higra_construction.higra_aligner import HiGraAligner
from higra_agent.retriever.higra_retriever.higra_retriever import HiGraRetriever
from higra_agent.retriever.higra_retriever.higra_schema import HierarchicalKnowledgeGraph

from higra_agent.llm_client.llm_client import LLMAsyncClient
from higra_agent.prompt.higra_construction_prompt import HiGraConstructionPrompt, ENTITY_LAYER_RESPONSE_FORMAT

class HiGraBuilder:
    def __init__(self, api_key, model_name, batch_size):
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size
    
    async def construct_entity_layer(
        self,
        document_path,
        entity_layer_path,
        entity_usage_path,
        max_tokens=15000
    ):
        ### Load data
        with open(document_path, 'r') as f:
          documents = json.load(f)
        
        ### Create data
        data = [
            (
                HiGraConstructionPrompt.system_prompt,
                HiGraConstructionPrompt.example_prompt+HiGraConstructionPrompt.user_prompt.format(
                    text=f"Title:{d['title']}\nContent:{d['content']}"
                )
            )
            for d in documents
        ]
        
        ### Call LLM
        client = LLMAsyncClient(
            api_key=self.api_key,
            model_name=self.model_name
        )
        
        response = await client.call_multiple_batched(
                data=data,
                batch_size=self.batch_size,
                response_format=ENTITY_LAYER_RESPONSE_FORMAT,
                max_tokens=max_tokens
            )
        
        ### Process Result
        result = [x[0] for x in response]
        usage  = [x[1] for x in response]
        er_data = [
            {
                'title': d['title'],
                'content': d['content'],
                'entity_layer': r,
            }
            for d, r in zip(documents, result)
        ]
        
        ### Save Data
        with open(entity_layer_path, "w", encoding='utf-8') as file:
            json.dump(er_data, file, indent=4, ensure_ascii=False)
        
        with open(entity_usage_path, "w", encoding='utf-8') as file:
            json.dump(usage, file, indent=4, ensure_ascii=False)

    async def debug_entity_layer(
        self,
        document_path: str,
        entity_layer_path: str,
        entity_usage_path: str,
        max_tokens: int = 15000,
    ):
        """
        Re-check entity_layer_path and re-run invalid entries.
        Invalid = entity_layer is not a dict.
        """
        # Load original documents
        with open(document_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
    
        # Load existing results
        with open(entity_layer_path, "r", encoding="utf-8") as f:
            er_data = json.load(f)
        with open(entity_usage_path, "r", encoding="utf-8") as f:
            usage = json.load(f)
    
        # Find bad indices
        bad_indices = [i for i, e in enumerate(er_data) if not isinstance(e.get("entity_layer"), dict)]
    
        if not bad_indices:
            print("✅ No errors found, all entity layers are dicts.")
            return
            
        print(f"⚠️ Found {len(bad_indices)} invalid entries. Re-running LLM calls...")
    
        # Prepare data for re-run
        data = [
            (
                HiGraConstructionPrompt.system_prompt,
                HiGraConstructionPrompt.example_prompt + HiGraConstructionPrompt.user_prompt.format(
                    text=f"Title:{documents[i]['title']}\nContent:{documents[i]['content']}"
                ),
            )
            for i in bad_indices
        ]
    
        # Init LLM client
        client = LLMAsyncClient(
            api_key=self.api_key,
            model_name=self.model_name,
        )
    
        # Call batched
        responses = await client.call_multiple_batched(
            data=data,
            batch_size=self.batch_size,
            response_format=ENTITY_LAYER_RESPONSE_FORMAT,
            max_tokens=max_tokens,
        )
    
        # Update results in place
        for idx, (res, usage_info) in zip(bad_indices, responses):
            er_data[idx]["entity_layer"] = res
            usage[idx] = usage_info
    
        # Save corrected data
        with open(entity_layer_path, "w", encoding="utf-8") as f:
            json.dump(er_data, f, indent=4, ensure_ascii=False)
        with open(entity_usage_path, "w", encoding="utf-8") as f:
            json.dump(usage, f, indent=4, ensure_ascii=False)
    
        print("✅ Finished repairing entity_layer file.")
        
    async def construct_open_higra(
        self,
        entity_layer_path,
        higra_save_path,
    ):  
        ModelCache.get_sentenizer()
        ModelCache.get_embedding_model()
        
        with open(entity_layer_path, "r", encoding="utf-8") as f:
            entity_layer_data = json.load(f)

        entity_layer_data = [
            {
                'paragraph_text': f"Title:{paragraph['title']}\nContent:{paragraph['content']}",
                'entity_layer': paragraph['entity_layer']
            }
            for paragraph in entity_layer_data
        ]
        
        higra_list = [
            HierarchicalKnowledgeGraph.from_dict(
                data,
                sentenizer=ModelCache._sentenizer
            ) for data in entity_layer_data
        ]
        
        higra_aligner = HiGraAligner(
            embedder=ModelCache._embedding_model,
            api_key=self.api_key,
            model_name=self.model_name,
            batch_size=self.batch_size
        )
        
        open_higra = await higra_aligner.build_open_higra(
            higra_list,
            save_path=higra_save_path
        )
        
        return open_higra
    
    def pre_calculate_embedding(
        self,
        higra_save_path,
        higra_embedding_save_path,
    ):
        ModelCache.get_tokenizer()
        ModelCache.get_embedding_model()

        if not os.path.exists(higra_embedding_save_path):
            os.makedirs(higra_embedding_save_path, exist_ok=True)
        
        with open(higra_save_path, "r", encoding='utf-8') as f:
            higra_data = json.load(f)

        higra = HierarchicalKnowledgeGraph.model_validate(higra_data)
        
        higra_retriever = HiGraRetriever(
            higra=higra,
            embedder=ModelCache._embedding_model,
            embedding_save_path=higra_embedding_save_path,
            api_key=self.api_key,
            model_name=self.model_name,
            config_dict=None
        )