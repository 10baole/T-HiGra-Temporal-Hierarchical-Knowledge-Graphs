import json
import yaml
import asyncio

from tqdm import tqdm
from higra_agent.higra_agent import HiGraAgent


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    config = load_config()

    higra_agent = HiGraAgent(
        higra_path=config['higra']['higra_path'],
        higra_embedding_path=config['higra']['embedding_path'],
        api_key=config['llm_client']['api_key'],
        model_name=config['llm_client']['model_name'],
        reasoning_mode=config['qa']['reasoning_mode'],
        retrieval_mode=config['qa']['retrieval_mode'],
        retrieval_config_dict=config['retrieval']
    )
    
    
    with open(config['higra']['dataset_path'], "r", encoding='utf-8') as file:
        dataset = json.load(file)

    async def process_dataset():
        result_list = []
        for index in tqdm(range(len(dataset))):
            try:
                question = dataset[index]["question"]
                prediction, usage, timing, history = await higra_agent.run(question)
                result_list.append(
                    {
                        "index": index,
                        "question": question,
                        "groundtruth": dataset[index]["answer"],
                        "prediction": prediction,
                        "history": history,
                        "usage": usage,
                        "timing": timing,
                    }
                )
            except Exception as e:
                print(e)
                result_list.append(
                    {
                        "index": index,
                        "question": dataset[index]["question"],
                        "groundtruth": dataset[index]["answer"],
                        "prediction": "error",
                        "history": None,
                        "usage": None,
                        "timing": None,
                    }
                )
        return result_list

    # Run async loop once
    result_list = asyncio.run(process_dataset())

    # Save results
    with open(config['qa']['pred_path'], "w", encoding="utf-8") as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()