import yaml
import os
import json
import sys
import asyncio
import argparse
import re
import string
from pathlib import Path
from collections import Counter
import codecs
from typing import List, Dict, Any, Optional

# Set UTF-8 encoding for stdout to handle special characters
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from higra_agent.retriever.higra_retriever.higra_schema import HierarchicalKnowledgeGraph
from higra_agent.retriever.higra_retriever.temporal_retriever import TemporalHiGraRetriever
from higra_agent.model_cache import ModelCache
from higra_agent.llm_client.llm_client import LLMAsyncClient
from higra_agent.prompt.extract_short_answer_prompt import LLMShortAnswerExtractionPrompt
from higra_agent.prompt.llm_as_a_judge_prompt import LLMEvaluatePrompt
from higra_agent.prompt.prediction_prompt import PredictionPrompt

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def create_extract_short_answer_message(predictions):
    return [
        (   
            LLMShortAnswerExtractionPrompt.system_prompt,
            LLMShortAnswerExtractionPrompt.user_prompt.format(
                question=prediction["question"], 
                prediction=prediction["prediction"]
            )
        ) for prediction in predictions
    ]

def create_evaluation_message(input_list):
    return [
        (   
            LLMEvaluatePrompt.system_prompt,
            LLMEvaluatePrompt.user_prompt.format(
                question=x["question"], 
                groundtruth=x["groundtruth"],
                prediction=x["prediction"],
            )
        ) for x in input_list
    ]

def check_answer_in_groundtruth(prediction: str, groundtruth_list: List[str]) -> bool:
    """Check if prediction contains any of the groundtruth answers"""
    normalized_pred = normalize_answer(prediction)
    
    for gt in groundtruth_list:
        normalized_gt = normalize_answer(gt)
        # Check if groundtruth is in prediction or prediction is in groundtruth
        if normalized_gt in normalized_pred or normalized_pred in normalized_gt:
            return True
    
    return False

def f1_score(prediction, ground_truth):
    """Calculates F1 score.

    Args:
        prediction: Predicted answer span (string).
        ground_truth: True answer span (string).

    Returns:
        F1 score.
    """
    try:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
    except Exception as e:
        print(e)
        return 0
        
    return f1

def load_temporal_kg(kg_path: str) -> HierarchicalKnowledgeGraph:
    """Load temporal knowledge graph from JSON"""
    with open(kg_path, 'r', encoding='utf-8') as f:
        kg_dict = json.load(f)
    
    # Reconstruct HKG from dict
    hkg = HierarchicalKnowledgeGraph.model_validate(kg_dict)
    return hkg


def load_qa_data(qa_path: str):
    """Load QA data"""
    with open(qa_path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def evaluate_temporal_rag(retriever, qa_data, llm_client):
    """Evaluate temporal RAG on QA data"""
    results = []
    
    for qa_item in qa_data:
        question = qa_item['question']
        
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")
     
        ground_truth = qa_item['answer']
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        
        print(f"Ground truth: {ground_truth}")
        
        result = await _evaluate_single_query(retriever, llm_client, question, ground_truth)
        results.append(result)
    
    return results

async def _evaluate_single_query(retriever, llm_client: LLMAsyncClient, question: str, ground_truth: List[str]) -> Dict[str, Any]:
    try:
        result = await retriever.retrieve_temporal(question=question)
        
        retrieved_passages = []
        ner_entities = []
        prediction = "Unknown"
        
        if result.get('should_refuse'):
            print(f"System refused: {result.get('reason')}")
            prediction = "Unknown"
        else:
            # Parse context from JSON string if needed
            context_str = result.get('context', '{}')
            context = json.loads(context_str) if isinstance(context_str, str) else context_str
            
            # Extract NER entities if available
            if 'ner_entities' in context:
                ner_entities = context['ner_entities']
            
            print(f"\nRetrieved context:")            
            if 'relevant_passages' in context:
                retrieved_passages = context['relevant_passages']
                print(f"  Retrieved {len(retrieved_passages)} passages")
                for i, passage in enumerate(retrieved_passages[:2], 1):
                    print(f"    Passage {i}: {passage[:100]}...")
            
            if 'ner_entities' in context and context['ner_entities']:
                print(f"  Extracted {len(context['ner_entities'])} NER entities")
                for entity in context['ner_entities'][:3]:
                    if isinstance(entity, dict):
                        print(f"    - {entity.get('name')} (type: {entity.get('type')})")
                    else:
                        print(f"    - {entity}")
            
            # Use LLM to generate answer from context
            if retrieved_passages:
                # Use more passages for better coverage
                num_passages = min(15, len(retrieved_passages))
                context_text = "\n\n".join(retrieved_passages[:num_passages])
                
                try:
                    # Use PredictionPrompt for structured temporal reasoning
                    response, _ = await llm_client.call_single(
                        system_prompt=PredictionPrompt.system_prompt,
                        user_prompt=PredictionPrompt.user_prompt.format(
                            context=context_text,
                            question=question
                        )
                    )
                    
                    # Handle None response from rate limiting
                    if response is None:
                        print(f"  ⚠️ LLM returned None (likely rate limited)")
                        prediction = "Unknown"
                    else:
                        response = response.strip()
                        
                        # Try to parse JSON response
                        try:
                            # Remove markdown code blocks if present
                            if response.startswith("```json"):
                                response = response[7:]
                            if response.startswith("```"):
                                response = response[3:]
                            if response.endswith("```"):
                                response = response[:-3]
                            response = response.strip()
                            
                            # Parse JSON
                            result = json.loads(response)
                            prediction = result.get("answer", "Unknown")
                            
                            # Log additional information if available
                            if "confidence" in result:
                                print(f"  Confidence: {result['confidence']}")
                            if "reasoning" in result and result["reasoning"]:
                                print(f"  Reasoning: {result['reasoning'][:100]}...")
                                
                        except json.JSONDecodeError:
                            # If JSON parsing fails, use the response as-is
                            print(f"  ⚠️ Could not parse JSON response, using raw text")
                            prediction = response
                        
                        # Debug: log if prediction is Unknown
                        if prediction.lower() == "unknown":
                            print(f"  ⚠️ LLM returned 'Unknown' - passages may not contain answer")
                        
                except Exception as e:
                    print(f"Error generating answer with LLM: {e}")
                    prediction = "Unknown"
            else:
                print(f"  ⚠️ No passages retrieved - cannot generate answer")
                prediction = "Unknown"
            
            print(f"\nPrediction: {prediction}")
        
        # Check if prediction contains any groundtruth answer
        has_answer = check_answer_in_groundtruth(prediction, ground_truth)
        
        return {
            "question": question,
            "groundtruth": ', '.join(ground_truth) if isinstance(ground_truth, list) else str(ground_truth),
            "groundtruth_list": ground_truth,  # Keep original list for checking
            "prediction": prediction,
            "retrieved_passages": retrieved_passages,
            "ner_entities": ner_entities,
            "has_answer_match": has_answer,  # Pre-check before LLM evaluation
            "result": result
        }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "question": question,
            "groundtruth": ', '.join(ground_truth) if isinstance(ground_truth, list) else str(ground_truth),
            "groundtruth_list": ground_truth,
            "prediction": "Error",
            "error": str(e),
            "retrieved_passages": [],
            "ner_entities": [],
            "has_answer_match": False
        }


def compute_metrics(results):
    total = len(results)
    
    # Count errors
    errors = sum(1 for r in results if 'error' in r)
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        return {
            "total": total,
            "errors": errors,
            "accuracy": 0,
            "f1": 0,
            "em": 0
        }
    
    # Calculate accuracy based on LLM evaluation (handle None evaluations)
    valid_evals = [x for x in valid_results if x.get('llm_evaluation') is not None and isinstance(x.get('llm_evaluation'), dict)]
    if valid_evals:
        accuracy = len([x for x in valid_evals if x.get('llm_evaluation', {}).get('predict') == True]) / len(valid_results)
    else:
        accuracy = 0
    
    # Calculate F1 scores
    f1_scores = [x.get('f1', 0) for x in valid_results]
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    # Calculate EM scores
    em_scores = [x.get('em', 0) for x in valid_results]
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
    
    return {
        "total": total,
        "errors": errors,
        "accuracy": accuracy,
        "f1": avg_f1,
        "em": avg_em
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Temporal RAG System on Temporal KG"
    )
    parser.add_argument(
        "--kg-path",
        type=str,
        default="./data/kg/temporal_kg.json",
        help="Path to temporal knowledge graph JSON (overrides config)"
    )
    parser.add_argument(
        "--qa-path",
        type=str,
        default="data/sample_data/qa.json",
        help="Path to QA data JSON"
    )
    parser.add_argument(
        "--embedding-path",
        type=str,
        default="./data/kg/temporal_embeddings",
        help="Path to pre-calculated embeddings (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/test_data",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    
    print("Evaluating Temporal RAG System...")
    print("="*80)
    
    # Load config first
    
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths from config or args (args override config)
    kg_path = args.kg_path
    embedding_path = args.embedding_path
    qa_path = args.qa_path
    output_dir = Path(args.output_dir)
    
    # Check if KG exists
    if not Path(kg_path).exists():
        print(f"ERROR: {kg_path} not found!")
        return
    
    # Load data
    print(f"Loading temporal knowledge graph from {kg_path}...")
    hkg = load_temporal_kg(str(kg_path))
    print(f"  Passages: {len(hkg.passage_layer.passages)}")
    print(f"  Nodes: {len(hkg.entity_layer.nodes)}")
    
    print(f"\nLoading QA data from {qa_path}...")
    qa_data = load_qa_data(str(qa_path))
    print(f"  QA items: {len(qa_data)}")
    
    qa_filename = Path(qa_path).name
    kg_filename = Path(kg_path).name
    
    print("\nInitializing retriever...")
    
    # Check if pre-calculated embeddings exist
    embedder = None
    embedding_save_path = None
    
    if embedding_path and Path(embedding_path).exists():
        print(f"  Pre-calculated embeddings found: {embedding_path}")
        embedder = ModelCache.get_embedding_model()
        embedding_save_path = str(embedding_path)
    else:
        print(f"  Loading embedder model (no pre-calculated embeddings)...")
        embedder = ModelCache.get_embedding_model()
    
    # Get API key from config or environment variable
    api_key = config['llm_client']['api_key']
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_KEY')
        if not api_key:
            print("WARNING: No API key found in config.yaml or environment variables.")
            print("Please set api_key in config.yaml or set OPENAI_API_KEY environment variable.")
            print("Some features requiring LLM will not work.")
    
    model_name = config['llm_client'].get('model_name', 'gpt-4o-mini')
    
    retriever = TemporalHiGraRetriever(
        higra=hkg,
        embedder=embedder,
        api_key=api_key,
        model_name=model_name,
        config_dict=config,
        embedding_save_path=embedding_save_path
    )
        
    # Initialize LLM client for evaluation
    print("\nInitializing LLM client for evaluation...")
    llm_client = LLMAsyncClient(
        api_key=api_key,
        model_name=model_name,
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = await evaluate_temporal_rag(retriever, qa_data, llm_client)
    
    # Extract short answers using LLM
    print("\nExtracting short answers...")
    batch_size = config['llm_client'].get('batch_size', 200)
    
    valid_results = [r for r in results if 'error' not in r]
    extract_short_answer_messages = create_extract_short_answer_message(valid_results)
    
    short_answers = await llm_client.call_multiple_batched(
        data=extract_short_answer_messages,
        batch_size=batch_size
    )
    
    # LLM as judge evaluation
    print("\nRunning LLM as judge evaluation...")
    evaluation_messages = create_evaluation_message(valid_results)
    
    evaluations = await llm_client.call_multiple_batched(
        data=evaluation_messages,
        response_format=LLMEvaluatePrompt.response_format,
        batch_size=batch_size
    )
    
    # Combine results with evaluations
    idx = 0
    for i, result in enumerate(results):
        if 'error' not in result:
            # Handle None responses from rate limiting
            short_answer = short_answers[idx][0] if short_answers[idx][0] is not None else "Unknown"
            llm_eval = evaluations[idx][0] if evaluations[idx][0] is not None else {"predict": False, "reasoning": "API error"}
            
            result['short_answer'] = short_answer
            result['short_answer_extraction_usage'] = short_answers[idx][1]
            result['llm_evaluation'] = llm_eval
            result['llm_evaluation_usage'] = evaluations[idx][1]
            
            # Override LLM evaluation if we have clear answer match
            if result.get('has_answer_match', False) and isinstance(result['llm_evaluation'], dict):
                result['llm_evaluation']['predict'] = True
            
            result['f1'] = f1_score(result['short_answer'], result['groundtruth'])
            idx += 1
        else:
            result['short_answer'] = "Error"
            result['llm_evaluation'] = {"predict": False}
            result['f1'] = 0
    
    # Compute metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    metrics = compute_metrics(results)
    print(f"\n📊 Metrics:")
    print(f"  Total queries: {metrics['total']}")
    print(f"  Errors: {metrics['errors']}")
    print(f"  Accuracy: {round(metrics['accuracy']*100, 1)}%")
    print(f"  F1: {round(metrics['f1']*100, 1)}%")
    print(f"  EM: {round(metrics['em']*100, 1)}%")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all results
    output_path = output_dir / "temporal_eval_results.json"
    print(f"\nSaving all results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False, default=str)
    
    incorrect_results = [
        r for r in results 
        if r.get('llm_evaluation') is not None 
        and isinstance(r.get('llm_evaluation'), dict) 
        and r.get('llm_evaluation', {}).get('predict') == False
    ]
    incorrect_output_path = output_dir / "incorrect_cases.json"
    print(f"Saving {len(incorrect_results)} incorrect cases to {incorrect_output_path}...")
    with open(incorrect_output_path, 'w', encoding='utf-8') as f:
        json.dump(incorrect_results, f, indent=4, ensure_ascii=False, default=str)
    
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())