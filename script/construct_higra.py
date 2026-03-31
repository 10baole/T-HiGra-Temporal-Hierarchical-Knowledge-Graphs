import os
import sys
import json
import yaml
import logging
import asyncio
import argparse

from tqdm import tqdm

from higra_agent.higra_construction.higra_builder import HiGraBuilder



def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def ensure_parent_dir(file_path: str):
    """
    Ensure the parent directory for a given file path exists.
    If not, create it recursively.
    """
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        
async def main():
    parser = argparse.ArgumentParser(description="HiGra pipeline runner")
    
    parser.add_argument(
        "--stage",
        type=str,
        nargs="+",
        default=["all"],
        choices=["er", "debug_er", "align", "embed", "all"],
        help="Which stage(s) to run: er (Entity Relationship Detection), "
             "align (Entity Alignment), embed (Precalculate Embedding), or all",
    )
    args = parser.parse_args()
    stages = args.stage

    config = load_config()

    coref_path = config['higra']['coref_document_path']    
    er_path = config['higra']['er_path']
    er_usage_path = config['higra']['er_usage_path']
    higra_path = config['higra']['higra_path']
    embedding_path = config['higra']['embedding_path']

    ensure_parent_dir(er_path)
    ensure_parent_dir(higra_path)
    ensure_parent_dir(embedding_path)
    
    higra_builder = HiGraBuilder(
        api_key=config['llm_client']['api_key'],
        model_name=config['llm_client']['model_name'],
        batch_size=config['llm_client']['batch_size'],
    )
    
    # Run stages
    if "all" in stages or "er" in stages:
        print("Stage: Entity Relationship Detection")
        await higra_builder.construct_entity_layer(
                document_path=coref_path,
                entity_layer_path=er_path,
                entity_usage_path=er_usage_path,
            )

    if "all" in stages or "debug_er" in stages:
        print("Stage: Entity Relationship Detection")
        await higra_builder.debug_entity_layer(
                document_path=coref_path,
                entity_layer_path=er_path,
                entity_usage_path=er_usage_path,
        )

    if "all" in stages or "align" in stages:
        print("Stage: Entity Alignment")
        await higra_builder.construct_open_higra(
                entity_layer_path=er_path,
                higra_save_path=higra_path,
            )

    if "all" in stages or "embed" in stages:
        print("Stage: Precalculate Embedding")
        higra_builder.pre_calculate_embedding(
            higra_save_path=higra_path,
            higra_embedding_save_path=embedding_path,
        )


if __name__ == "__main__":
    asyncio.run(main())