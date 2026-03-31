import re
import json
import yaml
import string
import asyncio
from collections import Counter

from higra_agent.llm_client.llm_client import LLMAsyncClient
from higra_agent.prompt.extract_short_answer_prompt import LLMShortAnswerExtractionPrompt
from higra_agent.prompt.llm_as_a_judge_prompt import LLMEvaluatePrompt



def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
        
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

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

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.

    Args:
        s: String to normalize.

    Returns:
        Cleaned string with lowercase, no punctuations, no articles, and
            and extraneous whitespace.
    """

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

def exact_match_score(prediction, ground_truth):
    """Calculates exact match (EM) score.

    Args:
        prediction: Predicted answer span (string).
        ground_truth: True answer span (string).

    Returns:
        EM score.
    """
    try:
        return (normalize_answer(prediction) == normalize_answer(ground_truth))
    except Exception as e:
        print(e)
        return 0

async def main():

    config = load_config()
    
    client = LLMAsyncClient(
        api_key=config['llm_client']['api_key'],
        model_name=config['llm_client']['model_name'],
    )
    
    batch_size = config['llm_client']['batch_size']
    dataset_path = config['higra']['dataset_path']
    pred_path = config['qa']['pred_path']
    eval_path = config['evaluation']['eval_path']
    
    
    dataset = load_json(dataset_path)
    prediction = load_json(pred_path)
    
    extract_short_answer_message_list = create_extract_short_answer_message(prediction)
    evaluation_message_list = create_evaluation_message(prediction)
    
    
    evaluation_list = await client.call_multiple_batched(
        data=evaluation_message_list, 
        response_format=LLMEvaluatePrompt.response_format, 
        batch_size=batch_size
    )
    
    short_answer_list = await client.call_multiple_batched(
        data=extract_short_answer_message_list,
        batch_size=batch_size
    )
    
    result = []
    for index in range(len(dataset)):
        prediction[index]['index'] = index
        prediction[index]['llm_evaluation'] = evaluation_list[index][0]
        prediction[index]['llm_evaluation_usage'] = evaluation_list[index][1]
        prediction[index]['short_answer'] = short_answer_list[index][0]
        prediction[index]['short_answer_extraction_usage'] = short_answer_list[index][1]
        prediction[index]['f1'] = f1_score(
            prediction[index]['short_answer'],
            prediction[index]['groundtruth']
        )
        prediction[index]['em'] = exact_match_score(
            prediction[index]['short_answer'],
            prediction[index]['groundtruth']
        )
        result.append(prediction[index])
    
    accuracy = len([x for x in result if x['llm_evaluation']['predict']==True]) / len(dataset)
    
    f1 = [x['f1'] for x in result]
    avg_f1 = sum(f1) / len(f1)
    
    em = [x['em'] for x in result]
    avg_em = sum(em) / len(em)
    
    print(f"==============")
    print(f"Accuracy: {round(accuracy*100, 1)}")
    print(f"F1: {round(avg_f1*100, 1)}")
    print(f"EM: {round(avg_em*100, 1)}")

    with open(eval_path, "w", encoding='utf-8') as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())