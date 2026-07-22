# T-HiGra: Temporal Reasoning over Hierarchical Knowledge Graphs

T-HiGra is a temporal-aware Retrieval-Augmented Generation (RAG) framework designed for **time-constrained open-domain question answering (ODQA)**. Unlike conventional Graph-RAG systems that focus only on semantic relevance, T-HiGra incorporates **temporal reasoning** into hierarchical knowledge graph retrieval, enabling the system to retrieve evidence that is both **semantically relevant** and **temporally consistent**.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API Key (or another supported LLM provider)
- Dependencies listed in `requirements.txt`

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/10baole/T-HiGra-Temporal-Hierarchical-Knowledge-Graphs.git
cd T-HiGra-Temporal-Hierarchical-Knowledge-Graphs
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY="your-openai-api-key"

LANGSMITH_TRACING="true"
LANGSMITH_API_KEY="your-langsmith-api-key"
```

> **Note:** `LANGSMITH_API_KEY` is optional and is only required when running the LLM-as-a-Judge evaluation.

---

# 💻 Usage

## 1. Build the Temporal Hierarchical Knowledge Graph

Construct the temporal hierarchical knowledge graph from the raw document collection.

```bash
python build_graph.py \
    --input_corpus data/timeqa_passages.json \
    --output_graph output/thigra_graph.json \
    --llm_model gpt-4o-mini
```

---

## 2. Run Temporal Retrieval & Question Answering

```python
from thigra.retriever import THiGraRetriever
from thigra.pipeline import QAPipeline

# Initialize retriever
retriever = THiGraRetriever(
    graph_path="output/thigra_graph.json",
    sigma=2.0,
    alpha=0.5
)

# Example query
query = "Who was the Prime Minister of the UK in October 2022?"

retrieved_context = retriever.retrieve(
    query=query,
    top_k=5
)

# Generate answer
pipeline = QAPipeline(retriever=retriever)

response = pipeline.answer(query)

print("Answer:", response["answer"])
print("Retrieved Passages:", response["documents"])
```

---

## 3. Evaluate the Framework

Evaluate retrieval and generation quality using the **LLM-as-a-Judge** protocol.

```bash
python evaluate.py \
    --dataset data/timeqa_test.json \
    --graph output/thigra_graph.json \
    --metrics correctness relevance groundedness retrieval_relevance
```

The evaluation reports the following metrics:

- Correctness
- Relevance
- Groundedness
- Retrieval Relevance

---

# 📂 Repository Structure

```text
T-HiGra-Temporal-Hierarchical-Knowledge-Graphs/
│
├── data/                          # Dataset loaders and TimeQA samples
│
├── thigra/
│   ├── construction/              # Graph construction pipeline
│   │   ├── coreference.py
│   │   ├── extraction.py
│   │   └── merge.py
│   │
│   ├── graph/                     # Hierarchical temporal graph representation
│   │
│   ├── retrieval/                 # Retrieval modules
│   │   ├── query_intent.py
│   │   ├── temporal_filter.py
│   │   ├── ppr.py
│   │   ├── adaptive_rrf.py
│   │   └── retriever.py
│   │
│   └── evaluation/                # LLM-as-a-Judge evaluation
│
├── build_graph.py                 # Offline graph construction
├── evaluate.py                    # Evaluation script
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

