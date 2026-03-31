import os
import sys
import json
import yaml
import torch
import re
from tqdm import tqdm
from typing import List, Tuple
from transformers import AutoTokenizer

import spacy
from fastcoref import spacy_component

# ------------------------------
# 1. Configuration Loading
# ------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)["higra"]

# ------------------------------
# 2. Dataset Preparation
# ------------------------------

def load_dataset(path: str) -> list:
    """Load dataset from JSON."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: list, path: str):
    """Save data to JSON with UTF-8 encoding."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# ------------------------------
# 3. Tokenizer and Chunking
# ------------------------------

TOKENIZER_CACHE = {}

def get_tokenizer():
    """Load and cache tokenizer."""
    if "tokenizer" not in TOKENIZER_CACHE:
        print("Loading tokenizer...")
        TOKENIZER_CACHE["tokenizer"] = AutoTokenizer.from_pretrained("bert-base-uncased")
    return TOKENIZER_CACHE["tokenizer"]


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=True))


def chunk_text_by_sentences(text: str, title: str, max_tokens: int = 512, overlap_sentences: int = 2) -> List[str]:
    tokenizer = get_tokenizer()
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk_sentences = []
    current_token_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = count_tokens(sentence, tokenizer)

         # If a single sentence exceeds max_tokens, add it as its own chunk
        if sentence_tokens > max_tokens:
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                merged_chunk = f"{title}\n{chunk_text}"
                chunks.append(merged_chunk)

                current_chunk_sentences = []
                current_token_count = 0

            # Add the long sentence as its own chunk
            merged_chunk = f"{title}\n{sentence}"
            chunks.append(merged_chunk)
            i += 1
            continue

        if current_token_count + sentence_tokens <= max_tokens:
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens
            i += 1
        else:
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                merged_chunk = f"{title}\n{chunk_text}"
                chunks.append(merged_chunk)

                overlap_start = max(0, len(current_chunk_sentences) - overlap_sentences)
                current_chunk_sentences = current_chunk_sentences[overlap_start:]
                current_token_count = count_tokens(" ".join(current_chunk_sentences), tokenizer)

                if current_token_count + sentence_tokens > max_tokens:
                    current_chunk_sentences = []
                    current_token_count = 0
            else:
                current_chunk_sentences = [sentence]
                current_token_count = sentence_tokens
                i += 1

    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        merged_chunk = f"{title}\n{chunk_text}"
        chunks.append(merged_chunk)
    
    return chunks


# ------------------------------
# 4. Coreference Resolution
# ------------------------------

MODEL_CACHE = {}

def get_coref_model():
    """Load and cache SpaCy FastCoref model, auto-selecting GPU if available."""
    if "coref" not in MODEL_CACHE:
        # Detect device automatically
        if torch.cuda.is_available():
            device = 0
            print("🚀 Using GPU for FastCoref.")
        else:
            device = -1
            print("💻 No GPU detected — using CPU for FastCoref.")

        print("Loading SpaCy FastCoref model...")
        model = spacy.load("en_core_web_lg")
        model.add_pipe("fastcoref", config={"device": device})
        MODEL_CACHE["coref"] = model

    return MODEL_CACHE["coref"]


def coreference_resolution(texts: List[str]) -> List[str]:
    """Apply coreference resolution to a list of texts."""
    model = get_coref_model()
    result = []
    doc_stream = model.pipe(
        texts, component_cfg={"fastcoref": {"resolve_text": True}}
    )

    for text in texts:
        try:
            doc = next(doc_stream)
            result.append(doc._.resolved_text)
        except Exception as e:
            print(f"[Warning] Coreference failed: {e}")
            result.append(text)

    return result


def process_chunks_batch(chunks: List[str], batch_size: int = 32) -> List[str]:
    resolved = []

    for i in tqdm(range(0, len(chunks), batch_size),
                  desc="Coref resolving chunks", unit="batch"):
        batch = chunks[i:i + batch_size]
        resolved.extend(coreference_resolution(batch))

    return resolved


# ------------------------------
# 5. Main Pipeline
# ------------------------------

def main():
    # Input and output paths
    input_path = "./data/dataset/processed/contexts.json"
    output_path = "./data/dataset/processed/final_context.json"

    print(f"Loading contexts from: {input_path}")
    contexts = load_dataset(input_path)

    print("Creating chunks from contexts...")
    all_chunks = []
    
    for ctx in tqdm(contexts, desc="Chunking contexts"):
        title = ctx.get("title", "")
        content = ctx.get("contexts", "")
        
        # Create chunks merged with title
        chunks = chunk_text_by_sentences(
            text=content,
            title=title,
            max_tokens=512,
            overlap_sentences=2
        )
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks from {len(contexts)} contexts")

    print("Starting coreference resolution on chunks...")
    resolved_chunks = process_chunks_batch(all_chunks, batch_size=256)

    # Format output as list of {"text": ...}
    output_data = [{"text": resolved_text} for resolved_text in resolved_chunks]

    save_json(output_data, output_path)
    print(f"Coreference-resolved chunks saved to: {output_path}")
    print(f"Total resolved chunks: {len(output_data)}")


if __name__ == "__main__":
    main()
