# T-HiGra-Temporal-Reasoning-over-Hierarchical-Knowledge-Graphs

🚀 Getting Started
Prerequisites
Python 3.10+

OpenAI API Key (or alternative LLM providers)

Dependencies listed in requirements.txt

Installation
Bash
# Clone the repository
git clone https://github.com/10baole/T-HiGra-Temporal-Hierarchical-Knowledge-Graphs.git
cd T-HiGra-Temporal-Hierarchical-Knowledge-Graphs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
Environment Configuration
Create a .env file in the root directory:

Đoạn mã
OPENAI_API_KEY="your-openai-api-key"
LANGSMITH_TRACING="true"
LANGSMITH_API_KEY="your-langsmith-api-key"  # Optional: For LLM-as-a-judge evaluation
💻 Usage
1. Build the Temporal Hierarchical Knowledge Graph
Construct the bi-layer temporal graph from raw text passages:

Bash
python build_graph.py \
    --input_corpus data/timeqa_passages.json \
    --output_graph output/thigra_graph.json \
    --llm_model gpt-4o-mini
2. Execute Temporal Retrieval & Question Answering
Run the temporal retriever and generate answers for time-constrained queries:

Python
from thigra.retriever import THiGraRetriever
from thigra.pipeline import QAPipeline

# Initialize T-HiGra Retriever
retriever = THiGraRetriever(
    graph_path="output/thigra_graph.json",
    sigma=2.0,
    alpha=0.5
)

# Run query
query = "Who was the Prime Minister of the UK in October 2022?"
retrieved_context = retriever.retrieve(query, top_k=5)

# Generate answer
pipeline = QAPipeline(retriever=retriever)
response = pipeline.answer(query)

print("Answer:", response["answer"])
print("Retrieved Passages:", response["documents"])
3. Evaluate Framework
Evaluate performance using the LLM-as-a-judge protocol (LangChain/LangSmith specification)[cite: 1, 4]:

Bash
python evaluate.py \
    --dataset data/timeqa_test.json \
    --graph output/thigra_graph.json \
    --metrics correctness relevance groundedness retrieval_relevance
