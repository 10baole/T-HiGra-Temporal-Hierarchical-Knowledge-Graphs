import json
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from higra_agent.llm_client.llm_client import LLMAsyncClient


SYSTEM_PROMPT = """You are an expert in temporal reasoning and question reformulation. Your task is to paraphrase questions to test specific temporal reasoning capabilities while preserving all original information and make sure that the paraphrased questions remain clear and unambiguous with the same meaning and the answer."""

AUGMENTATION_PROMPTS = {
    "cognition": """Paraphrase this question to test COGNITION - the ability to retrieve knowledge at specific timestamps or temporal intervals.

Original question: {question}
Available timestamps: {timestamps}

Rules:
- TIMESTAMP SUBTASK: Transform to ask about a SPECIFIC DATE within available intervals
  Example: "Who was President from 2001-2009?" → "Who was President on January 20, 2005?"
  
- INTERVAL SUBTASK: Transform to ask about a FULL TIME PERIOD
  Example: "Who was President in 2005?" → "Who served as President from 2001 to 2009?"

- Randomly choose between timestamp or interval format
- Use precise date formats (DD Month YYYY) for timestamps
- Use clear interval boundaries [Tstart, Tend] for intervals
- Preserve all entities and maintain factual accuracy
- The answer must remain exactly the same as the original

Return ONLY the paraphrased question, nothing else.""",

    "awareness": """Paraphrase this question to test AWARENESS - the ability to detect and handle temporal misalignment between context and query.

Original question: {question}
Available timestamps: {timestamps}

Rules:
- FUTURE MISALIGNED CONTEXT (F.M.C.): Add contemporary/recent context that is NEWER than the query timestamp
  Example: "Who was President in 2019?" → "Given that Joe Biden is currently the President in 2024, who was President back in 2019?"
  
- PAST MISALIGNED CONTEXT (P.M.C.): Add historical context that is OLDER than the query timestamp
  Example: "Who is President now?" → "Considering that Barack Obama was President from 2009-2017, who is the President as of 2024?"

- Create temporal tension by mixing different time references
- Test if the system can distinguish and properly use/ignore misaligned information
- Keep the core question and entities intact

Return ONLY the paraphrased question, nothing else.""",

    "trustworthiness": """Paraphrase this question to test TRUSTWORTHINESS - the ability to refuse unanswerable questions with invalid timestamps.

Original question: {question}
Available timestamps: {timestamps}

Rules:
- PAST UNANSWERABLE: Ask about dates BEFORE the earliest historical record
  Example: "Who was President of the United States in 1750?" (before USA existed)
  
- FUTURE UNANSWERABLE: Ask about dates FAR IN THE FUTURE (e.g., after 2050)
  Example: "Who will be President of the United States on December 31, 2050?"

- The timestamp should be clearly outside valid historical/reasonable future range
- System should respond "Unknown" or refuse to answer
- Preserve entities but make timestamp deliberately invalid
- Test boundary recognition (before earliest record or beyond reasonable future)

Return ONLY the paraphrased question, nothing else.""",

    "understanding": """Paraphrase this question to test UNDERSTANDING - the ability to understand explicit and implicit temporal concepts.

Original question: {question}
Available timestamps: {timestamps}

Rules:
- EXPLICIT TEMPORAL CONCEPT (E.T.C.): Use clear date expressions (DD Month YYYY or [Tstart, Tend])
  Example: "Who was President during Obama's term?" → "Who was President from 20 January 2009 to 20 January 2017?"
  
- IMPLICIT TEMPORAL CONCEPT (I.T.C.): Replace dates with semantic/historical markers
  Example: "Who was President in 2010?" → "Who was President when Barack Obama was in office?"
  Example: "Who was X from 2017-2021?" → "Who was X during the Trump administration?"

- Use event-based temporal anchors: "during [major event]", "when [person] was [role]"
- Avoid ambiguous events (e.g., events that occurred multiple times)
- Test if system can map semantic descriptions to actual time intervals
- The implicit reference should uniquely identify a time period

Return ONLY the paraphrased question, nothing else.""",

    "reasoning": """Paraphrase this question to test REASONING - the ability to perform temporal ranking and calculation.

Original question: {question}
Available timestamps: {timestamps}

Rules:
- RANKING (R.K.) SUBTASK: Compare temporal order of TWO events
  Example: "Who was President in 2001 and 2009?" → "Who became President first: George W. Bush or Barack Obama? When did each serve?"
  Example: "List Presidents from 2000-2020" → "Which came earlier: the Obama administration or the Trump administration?"
  
- CALCULATION (C.A.L.) SUBTASK: Require date arithmetic to find answer
  Example: "Who was President in 2013?" → "If Obama became President on January 20, 2009, who was President 1,461 days later?"
  Example: "Who was X in 2020?" → "Exactly 5 years after Trump took office, who held position X?"

- Require explicit temporal reasoning (comparison, ordering, duration calculation)
- Include multiple time references or time differences
- Force system to compute dates and retrieve corresponding attributes

Return ONLY the paraphrased question, nothing else."""
}


async def augment_questions(
    qa_data: List[Dict],
    augmentation_type: str,
    client: LLMAsyncClient,
    batch_size: int = 50
) -> List[Dict]:
    """
    Augment questions using LLM batch processing
    
    Args:
        qa_data: List of QA items from temporal_qa_data.json
        augmentation_type: One of: cognition, awareness, trustworthiness, understanding, reasoning
        client: LLM client for API calls
        batch_size: Number of requests per batch
    
    Returns:
        Augmented QA data with paraphrased questions
    """
    prompt_template = AUGMENTATION_PROMPTS[augmentation_type]
    
    # Prepare batch of messages
    message_list = []
    for item in qa_data:
        question = item['question']
        timestamps = [td['timestamp'] for td in item['temporal_data']]
        
        user_prompt = prompt_template.format(
            question=question,
            timestamps=", ".join(timestamps)
        )
        
        message_list.append((SYSTEM_PROMPT, user_prompt))
    
    print(f"\n{'='*80}")
    print(f"Augmenting {len(qa_data)} questions for type: {augmentation_type.upper()}")
    print(f"{'='*80}")
    
    # Call LLM in batches
    results = await client.call_multiple_batched(
        data=message_list,
        batch_size=batch_size,
        response_format=None,  # Plain text response
        retries=2,
        temperature=0.7,  # Slightly higher for creativity
        max_tokens=200
    )
    
    # Process results
    augmented_data = []
    success_count = 0
    fail_count = 0
    
    for i, (item, (result, usage)) in enumerate(zip(qa_data, results)):
        if result and not isinstance(result, Exception):
            paraphrased_question = result.strip()
            
            augmented_item = {
                "question": paraphrased_question,
                "temporal_data": item['temporal_data'],  # Preserve original temporal_data structure
                "original_question": item['question'],
                "augmentation_type": augmentation_type
            }
            
            augmented_data.append(augmented_item)
            success_count += 1
            
            if i < 3:  # Show first 3 examples
                print(f"\nExample {i+1}:")
                print(f"  Original: {item['question']}")
                print(f"  Augmented: {paraphrased_question}")
        else:
            print(f"\nFailed to augment question {i+1}: {item['question']}")
            print(f"  Error: {result}")
            
            # Keep original on failure with proper structure
            fallback_item = {
                "question": item['question'],
                "temporal_data": item['temporal_data'],
                "original_question": item['question'],
                "augmentation_type": augmentation_type,
                "augmentation_failed": True
            }
            
            augmented_data.append(fallback_item)
            fail_count += 1
    
    print(f"\n{'='*80}")
    print(f"Augmentation complete: {success_count} success, {fail_count} failed")
    print(f"{'='*80}\n")
    
    return augmented_data


async def main():
    """Main function to augment temporal QA data"""
    
    # Paths
    input_file = "data/sample_higra_data/document/filtered-coref-documents.json"
    
    # Load config for API key
    config_path = project_root / "config.yaml"
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize LLM client
    client = LLMAsyncClient(
        api_key=config['llm_client']['api_key'],
        model_name=config['llm_client'].get('model_name', 'gpt-4o-mini')
    )
    
    # Load temporal QA data
    print(f"Loading data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    print(f"Loaded {len(qa_data)} QA items")
    
    # Augmentation types
    augmentation_types = [
        "cognition",
        "awareness",
        "trustworthiness", 
        "understanding",
        "reasoning"
    ]
    
    # Generate augmented data for each type
    for aug_type in augmentation_types:
        print(f"\n{'#'*80}")
        print(f"Processing: {aug_type.upper()}")
        print(f"{'#'*80}")
        
        augmented_data = await augment_questions(
            qa_data=qa_data,
            augmentation_type=aug_type,
            client=client,
            batch_size=50
        )
        
        # Save augmented data
        output_file = f"data/augmented_data/data_{aug_type}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved augmented data to: {output_file}")
        print(f"Total items: {len(augmented_data)}")
    
    print(f"\n{'#'*80}")
    print("All augmentations complete!")
    print(f"{'#'*80}")
    print("\nGenerated files:")
    for aug_type in augmentation_types:
        output_file = f"data/augmented_data/data_{aug_type}.json"
        print(f"  - {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
