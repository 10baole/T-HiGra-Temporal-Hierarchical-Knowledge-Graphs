import os
import json
import sys
import yaml
import asyncio
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from higra_agent.retriever.higra_retriever.higra_schema import (
    HierarchicalKnowledgeGraph, PassageLayer, SentenceLayer, EntityLayer,
    Passage, Sentence, Node, Edge, TemporalInterval
)
from higra_agent.retriever.higra_retriever.temporal_utils import TemporalNormalizer
from higra_agent.llm_client.llm_client import LLMAsyncClient
from higra_agent.prompt.higra_construction_prompt import HiGraConstructionPrompt, ENTITY_LAYER_RESPONSE_FORMAT
from higra_agent.model_cache import ModelCache


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_temporal_qa_data(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def parse_temporal_from_entity_layer(node_data: dict) -> TemporalInterval:
    valid_time = node_data.get('valid_time')
    
    if not valid_time:
        return None
    
    start = valid_time.get('start', '')
    end = valid_time.get('end', '')
    
    if start and start.strip():
        return TemporalInterval(start=start, end=end if end and end.strip() else None)
    
    return None


async def build_temporal_kg_llm(
    qa_data,
    api_key: str,
    model_name: str,
    batch_size: int,
    max_tokens: int = 15000,
):

    temporal_normalizer = TemporalNormalizer()
    sentenizer = ModelCache.get_sentenizer()
    
    all_passages = []
    all_sentences = []
    all_nodes = []
    all_edges = []
    
    node_id_counter = 1
    edge_id_counter = 1
    
    llm_inputs = []
    contexts = []  
    
    for qa_item in qa_data:
        context_text = qa_item.get('text', '')
        if not context_text:
            continue
        
        user_prompt = HiGraConstructionPrompt.example_prompt + HiGraConstructionPrompt.user_prompt.format(text=context_text)
        llm_inputs.append((HiGraConstructionPrompt.system_prompt, user_prompt))
        
        contexts.append(context_text)
    
    # Batch call LLM
    print(f"Calling LLM for {len(llm_inputs)} contexts...")
    client = LLMAsyncClient(api_key=api_key, model_name=model_name)
    
    llm_responses = await client.call_multiple_batched(
        data=llm_inputs,
        batch_size=batch_size,
        response_format=ENTITY_LAYER_RESPONSE_FORMAT,
        max_tokens=max_tokens
    )
    
    print(f"Processing {len(llm_responses)} LLM responses...")
    
    for idx, (entity_layer, usage_info) in enumerate(llm_responses):
        context_text = contexts[idx]
        
        if not isinstance(entity_layer, dict) or 'nodes' not in entity_layer:
            print(f"Warning: Invalid entity layer at index {idx}, skipping. Response type: {type(entity_layer)}, content: {str(entity_layer)[:200]}")
            continue
        
        passage_id = f"passage_{len(all_passages) + 1}"
        
        # Sentenize text
        doc = sentenizer(context_text)
        sentences = []
        for sent in doc.sents:
            sent_id = f"sent_{len(all_sentences) + 1}"
            sentences.append(Sentence(
                sentence_id=sent_id,
                passage_id=passage_id,
                text=sent.text.strip(),
                node_map=[]
            ))
        
        all_sentences.extend(sentences)
        
        # Determine primary_time from LLM-extracted temporal entities or fallback
        primary_time = None
        for node_data in entity_layer.get('nodes', []):
            temporal_interval = parse_temporal_from_entity_layer(node_data)
            if temporal_interval:
                primary_time = temporal_interval
                break
                
        # Create passage
        passage = Passage(
            passage_id=passage_id,
            text=context_text,
            primary_time=primary_time,
            sentence_map=[s.sentence_id for s in sentences],
            node_map=[]
        )
        all_passages.append(passage)
        
        # Create nodes from LLM extraction
        node_id_map = {}  # Map from LLM node id to our node id
        
        for node_data in entity_layer.get('nodes', []):
            node_id = f"node_{node_id_counter}"
            node_id_counter += 1
            
            valid_time = parse_temporal_from_entity_layer(node_data)
            
            if not valid_time:
                valid_time = primary_time
            
            node = Node(
                id=node_id,
                name=node_data.get('name', ''),
                type=node_data.get('type', ['Entity']),
                aliases=node_data.get('aliases', []),
                description=node_data.get('description', ''),
                valid_time=valid_time,
                properties={
                    **node_data.get('properties', {}),
                    'llm_extracted': True
                }
            )
            all_nodes.append(node)
            passage.node_map.append(node_id)
            
            # Map LLM node id to our node id
            node_id_map[node_data.get('id')] = node_id
            
            # Link sentences to node
            node_name_lower = node.name.lower()
            for sent in sentences:
                if node_name_lower in sent.text.lower():
                    sent.node_map.append(node_id)
        
        # Create edges from LLM extraction
        for edge_data in entity_layer.get('edges', []):
            llm_source_id = edge_data.get('source_node_id')
            llm_target_id = edge_data.get('target_node_id')
            
            # Map to our node ids
            source_id = node_id_map.get(llm_source_id)
            target_id = node_id_map.get(llm_target_id)
            
            if not source_id or not target_id:
                continue
            
            # Parse edge temporal info
            edge_properties = edge_data.get('properties', {})
            edge_valid_time = None
            if 'valid_time' in edge_properties:
                vt = edge_properties['valid_time']
                if vt and vt.get('start'):
                    edge_valid_time = TemporalInterval(
                        start=vt.get('start'),
                        end=vt.get('end')
                    )
            
            if not edge_valid_time:
                edge_valid_time = primary_time
            
            edge = Edge(
                edge_id=f"edge_{edge_id_counter}",
                source_node_id=source_id,
                target_node_id=target_id,
                relationship_name=edge_data.get('relationship_name', 'related_to'),
                valid_time=edge_valid_time,
                properties={
                    **edge_properties,
                    'llm_extracted': True
                }
            )
            all_edges.append(edge)
            edge_id_counter += 1
    
    hkg = HierarchicalKnowledgeGraph(
        passage_layer=PassageLayer(passages=all_passages),
        sentence_layer=SentenceLayer(sentences=all_sentences),
        entity_layer=EntityLayer(nodes=all_nodes, edges=all_edges)
    )
    
    return hkg


async def main():

    config_path = "config.yaml"
    data_path = "data/sample_data/contexts.json"
    er_path = "./data/kg/temporal_entity_layer.json"
    er_usage_path = "./data/kg/temporal_entity_usage.json"
    higra_path = "./data/kg/temporal_kg.json"
    llm_extracted_path = "./data/kg/llm_extracted_entities.json"
    embedding_path = "./data/kg/temporal_embeddings"

    config = load_config(config_path)
    for path in [er_path, er_usage_path, higra_path, llm_extracted_path, embedding_path]:
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
    
    print("Building temporal knowledge graph with LLM...")
    print(f"Data source: {data_path}")
    qa_data = load_temporal_qa_data(data_path)
    print(f"Loaded {len(qa_data)} QA items")
    
    print("Loading spaCy sentenizer...")
    ModelCache.get_sentenizer()
    
    print("Calling LLM to extract temporal entities and relationships...")
    
    hkg = await build_temporal_kg_llm(
        qa_data=qa_data,
        api_key=config['llm_client']['api_key'],
        model_name=config['llm_client']['model_name'],
        batch_size=config['llm_client']['batch_size'],
    )
    
    print("\n" + "="*60)
    print("Temporal Knowledge Graph Statistics")
    print("="*60)
    print(f"Total passages: {len(hkg.passage_layer.passages)}")
    print(f"Total sentences: {len(hkg.sentence_layer.sentences)}")
    print(f"Total nodes: {len(hkg.entity_layer.nodes)}")
    print(f"Total edges: {len(hkg.entity_layer.edges)}")
    
    # Extract timestamps
    context_time = set()
    for passage in hkg.passage_layer.passages:
        if passage.primary_time:
            context_time.add(passage.primary_time)
    for node in hkg.entity_layer.nodes:
        if node.valid_time:
            context_time.add(node.valid_time)
    
    # Convert TemporalInterval objects to dicts for JSON serialization
    context_time_list = [t.to_dict() for t in sorted(list(context_time))]
    
    
    # Save entity layer (for construct_higra.py compatibility)
    if er_path:
        print(f"\nSaving entity layer to {er_path}...")
        entity_layer_dict = {
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "type": n.type,
                    "aliases": n.aliases,
                    "description": n.description,
                    "valid_time": n.valid_time.to_dict() if n.valid_time else None,
                    "is_temporal_entity": n.is_temporal_entity,
                    "properties": n.properties
                }
                for n in hkg.entity_layer.nodes
            ],
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "source_node_id": e.source_node_id,
                    "target_node_id": e.target_node_id,
                    "relationship_name": e.relationship_name,
                    "valid_time": e.valid_time.to_dict() if e.valid_time else None,
                    "temporal_relation": e.temporal_relation,
                    "properties": e.properties
                }
                for e in hkg.entity_layer.edges
            ]
        }
        with open(er_path, 'w', encoding='utf-8') as f:
            json.dump(entity_layer_dict, f, indent=2, ensure_ascii=False)
    
    # Save usage info (token usage statistics)
    if er_usage_path:
        print(f"Saving usage info to {er_usage_path}...")
        usage_info = {
            "total_nodes": len(hkg.entity_layer.nodes),
            "total_edges": len(hkg.entity_layer.edges),
            "total_passages": len(hkg.passage_layer.passages),
            "total_sentences": len(hkg.sentence_layer.sentences),
            "context_time": context_time_list,
            "build_method": "LLM-based extraction",
            "model": config['llm_client']['model_name']
        }
        with open(er_usage_path, 'w', encoding='utf-8') as f:
            json.dump(usage_info, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaving full knowledge graph to {higra_path}...")
    
    kg_dict = {
        "passage_layer": {
            "passages": [
                {
                    "passage_id": p.passage_id,
                    "text": p.text,
                    "primary_time": p.primary_time.to_dict() if p.primary_time else None,
                    "node_map": p.node_map,
                    "sentence_map": p.sentence_map
                }
                for p in hkg.passage_layer.passages
            ]
        },
        "sentence_layer": {
            "sentences": [
                {
                    "sentence_id": s.sentence_id,
                    "passage_id": s.passage_id,
                    "text": s.text,
                    "node_map": s.node_map
                }
                for s in hkg.sentence_layer.sentences
            ]
        },
        "entity_layer": {
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "type": n.type,
                    "aliases": n.aliases,
                    "description": n.description,
                    "valid_time": n.valid_time.to_dict() if n.valid_time else None,
                    "is_temporal_entity": n.is_temporal_entity,
                    "properties": n.properties
                }
                for n in hkg.entity_layer.nodes
            ],
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "source_node_id": e.source_node_id,
                    "target_node_id": e.target_node_id,
                    "relationship_name": e.relationship_name,
                    "valid_time": e.valid_time.to_dict() if e.valid_time else None,
                    "temporal_relation": e.temporal_relation,
                    "properties": e.properties
                }
                for e in hkg.entity_layer.edges
            ]
        }
    }
    
    with open(higra_path, 'w', encoding='utf-8') as f:
        json.dump(kg_dict, f, indent=2, ensure_ascii=False)
    
    # Save LLM-extracted entities and relationships separately
    if llm_extracted_path:
        print(f"\nSaving LLM-extracted entities to {llm_extracted_path}...")
        llm_extracted_nodes = [
            {
                "id": n.id,
                "name": n.name,
                "type": n.type,
                "aliases": n.aliases,
                "description": n.description,
                "valid_time": n.valid_time.to_dict() if n.valid_time else None,
                "is_temporal_entity": n.is_temporal_entity,
                "properties": n.properties
            }
            for n in hkg.entity_layer.nodes
            if n.properties.get('llm_extracted', False)
        ]
        
        llm_extracted_edges = [
            {
                "edge_id": e.edge_id,
                "source_node_id": e.source_node_id,
                "target_node_id": e.target_node_id,
                "relationship_name": e.relationship_name,
                "valid_time": e.valid_time.to_dict() if e.valid_time else None,
                "temporal_relation": e.temporal_relation,
                "properties": e.properties
            }
            for e in hkg.entity_layer.edges
            if e.properties.get('llm_extracted', False)
        ]
        
        llm_extracted_data = {
            "metadata": {
                "build_method": "LLM-based extraction",
                "model": config['llm_client']['model_name'],
                "total_nodes": len(llm_extracted_nodes),
                "total_edges": len(llm_extracted_edges),
                "extraction_date": "2026-02-03"
            },
            "nodes": llm_extracted_nodes,
            "edges": llm_extracted_edges
        }
        
        with open(llm_extracted_path, 'w', encoding='utf-8') as f:
            json.dump(llm_extracted_data, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved {len(llm_extracted_nodes)} LLM-extracted nodes")
        print(f"  Saved {len(llm_extracted_edges)} LLM-extracted edges")
    
    # Save embeddings (pre-calculate for retriever)
    if embedding_path:
        print(f"\nPre-calculating embeddings and saving to {embedding_path}...")
        embedder = ModelCache.get_embedding_model()
        
        # Create embedding directory structure
        os.makedirs(embedding_path, exist_ok=True)
        
        # Save passage embeddings (for dense retriever)
        print(f"  Embedding passages...")
        passage_texts = [p.text for p in hkg.passage_layer.passages]
        if passage_texts:
            passage_embeddings = embedder.encode(passage_texts, show_progress_bar=True)
            
            passage_dir = os.path.join(embedding_path, 'passage_dense')
            os.makedirs(passage_dir, exist_ok=True)
            
            import numpy as np
            np.save(os.path.join(passage_dir, 'embeddings.npy'), passage_embeddings)
            # Save mapping (index -> passage_id)
            passage_mapping = [p.passage_id for p in hkg.passage_layer.passages]
            np.save(os.path.join(passage_dir, 'mapping.npy'), np.array(passage_mapping))
            print(f"  Saved {len(passage_embeddings)} passage embeddings")
        
        # Save sentence embeddings (for sentence retriever)
        print(f"  Embedding sentences...")
        sentence_texts = [s.text for s in hkg.sentence_layer.sentences]
        if sentence_texts:
            sentence_embeddings = embedder.encode(sentence_texts, show_progress_bar=True)
            
            sentence_dir = os.path.join(embedding_path, 'sentence_dense')
            os.makedirs(sentence_dir, exist_ok=True)
            
            np.save(os.path.join(sentence_dir, 'embeddings.npy'), sentence_embeddings)
            sentence_mapping = [s.sentence_id for s in hkg.sentence_layer.sentences]
            np.save(os.path.join(sentence_dir, 'mapping.npy'), np.array(sentence_mapping))
            print(f"  Saved {len(sentence_embeddings)} sentence embeddings")
    
    print("\n" + "="*60)
    print("Build Complete!")
    print("="*60)
    print(f"Entity Layer: {er_path}")
    print(f"Usage Info: {er_usage_path}")
    print(f"Full KG: {higra_path}")
    print(f"LLM Extracted: {llm_extracted_path}")
    print(f"Embeddings: {embedding_path}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())