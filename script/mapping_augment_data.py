"""
Script to create awareness_2019.json file with structure matching valid_question_2019.json
Mapping answers from valid_question_2019.json to answers with timestamp 2019 in data_awareness.json
Questions are taken from data_awareness.json based on the mappings found
"""

import json
from pathlib import Path
from typing import List, Dict, Set
import sys

def load_json(filepath: str) -> any:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: any, filepath: str):
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    return answer.lower().strip()

def find_matching_awareness_questions(
    valid_answers: List[str], 
    awareness_data: List[Dict]
) -> List[Dict]:
    """
    Find questions from awareness data that have matching answers in timestamp 2019
    
    Args:
        valid_answers: List of answers from valid_question_2019.json
        awareness_data: All data from data_awareness.json
    
    Returns:
        List of matched questions with their 2019 temporal data
    """
    # Normalize valid answers for comparison
    normalized_valid_answers = {normalize_answer(ans) for ans in valid_answers}
    
    matched_questions = []
    
    for item in awareness_data:
        question = item['question']
        temporal_data_list = item.get('temporal_data', [])
        
        # Find 2019 temporal data
        temporal_2019 = None
        for temporal in temporal_data_list:
            if temporal.get('timestamp') == '2019':
                temporal_2019 = temporal
                break
        
        if not temporal_2019:
            continue
        
        # Get ground truth answers for 2019
        ground_truth_answers = temporal_2019.get('ground_truth_answers', [])
        
        # Check if any answer matches
        normalized_awareness_answers = {normalize_answer(ans) for ans in ground_truth_answers}
        
        # Find intersection
        matching_answers = normalized_valid_answers & normalized_awareness_answers
        
        if matching_answers:
            # Found a match! Use the original answers from awareness data
            matched_questions.append({
                'question': question,
                'answer': ground_truth_answers,
                'matched_count': len(matching_answers)
            })
    
    return matched_questions

def main():
    # Define paths
    base_path = Path(__file__).parent.parent
    valid_question_path = base_path / 'data' / 'sample_data' / 'valid_question_2019.json'
    awareness_path = base_path / 'data' / 'augmented_data' / 'data_awareness.json'
    output_path = base_path / 'data' / 'sample_data' / 'awareness_2019.json'
    
    print("Loading data files...")
    
    # Load data
    valid_questions = load_json(valid_question_path)
    awareness_data = load_json(awareness_path)
    
    print(f"Loaded {len(valid_questions)} questions from valid_question_2019.json")
    print(f"Loaded {len(awareness_data)} questions from data_awareness.json")
    
    # Collect all answers from valid_question_2019.json
    all_valid_answers = []
    for item in valid_questions:
        all_valid_answers.extend(item['answer'])
    
    print(f"Total valid answers: {len(all_valid_answers)}")
    print(f"Unique valid answers: {len(set(normalize_answer(a) for a in all_valid_answers))}")
    
    # Find matching questions
    print("\nFinding matching questions...")
    matched_questions = find_matching_awareness_questions(all_valid_answers, awareness_data)
    
    print(f"\nFound {len(matched_questions)} matching questions")
    
    # Remove the matched_count field before saving
    output_data = [{'question': q['question'], 'answer': q['answer']} 
                   for q in matched_questions]
    
    # Save output
    save_json(output_data, output_path)
    print(f"\nSaved {len(output_data)} questions to {output_path}")
    
    # Print some statistics
    if matched_questions:
        print("\nSample matched questions (first 3):")
        for i, q in enumerate(matched_questions[:3], 1):
            print(f"\n{i}. Question: {q['question']}")
            print(f"   Answers: {q['answer'][:5]}...")  # Show first 5 answers
            print(f"   Matched answers: {q['matched_count']}")

if __name__ == '__main__':
    main()
