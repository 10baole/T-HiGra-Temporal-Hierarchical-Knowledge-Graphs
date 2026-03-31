import json
from pathlib import Path

def filter_data_by_questions(data_original_path, valid_questions_path, output_path):

    print(f"Loading valid questions from {valid_questions_path}...")
    with open(valid_questions_path, 'r', encoding='utf-8') as f:
        valid_questions_data = json.load(f)
    
    # Extract questions from valid_question_2405.json
    valid_questions = {item['question'] for item in valid_questions_data}
    print(f"Found {len(valid_questions)} valid questions")
    
    # Load data_original.json
    print(f"Loading data from {data_original_path}...")
    with open(data_original_path, 'r', encoding='utf-8') as f:
        data_original = json.load(f)
    
    print(f"Total entries in data_original: {len(data_original)}")
    
    # Filter data based on matching questions
    filtered_data = []
    for entry in data_original:
        question = entry.get('question', '')
        if question in valid_questions:
            # Extract contexts from temporal_data
            contexts = []
            temporal_data = entry.get('temporal_data', [])
            
            i = 0
            for temp_entry in temporal_data:
                if 'context' in temp_entry:
                    filtered_data.append({'context': temp_entry['context']})
                i += 1
                if i == 3: break
    
    print(f"Filtered entries: {len(filtered_data)}")
    
    # Save filtered data
    print(f"Saving filtered data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"Done! Saved {len(filtered_data)} entries to {output_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    data_original_path = base_dir / "data" / "augmented_data" / "data_original.json"
    valid_questions_path = base_dir / "data" / "sample_data" / "valid_question_2405.json"
    output_path = base_dir / "data" / "sample_data" / "data_2405.json"
    filter_data_by_questions(data_original_path, valid_questions_path, output_path)