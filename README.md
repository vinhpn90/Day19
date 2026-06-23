# Evaluation of GraphRAG vs Flat RAG (U.S. EV Market Lab)

This repository implements and evaluates a **GraphRAG** pipeline (using **NetworkX** for graph traversal) against a **Flat RAG** pipeline (using **FAISS** and **Sentence Transformers** for dense vector retrieval) on the **Tech Company Corpus** (70 research and market report documents regarding the U.S. EV industry).

The project is configured to work with an OpenAI-compatible reasoning model endpoint (`/var/lib/vllm/hf/gpt-oss-120b`).

---

## Key Features

- **Document Processing**: Cleans binary PDF stream garbage (e.g. from `doc_50.txt`) before RAG indexing.
- **Triples Indexing**: Extracts knowledge graph triples `(Subject, Relation, Object)` using parallel processing and caches them locally to save API tokens and time.
- **FAISS Flat RAG**: Chunks documents and builds a dense vector search index utilizing `faiss.IndexFlatIP` and the local `all-MiniLM-L6-v2` embedding model.
- **Optimized GraphRAG**: 
  - Dynamic entity query matching.
  - **Hub-node mitigation** (stops BFS traversal through high-degree hub nodes like `"U.S."` or `"Q1 2024"` to prevent context cluttering).
  - **Brand connection linking** (`BRAND_LINK` edges are automatically added between same-brand entities to connect disjoint components).
- **Comparison Benchmark**: Evaluates both systems on 20 complex, multi-hop industry questions, tracking response time, accuracy, token count, and cost.

---

## Project Structure

```
├── dataset/                  # Directory containing 70 txt documents
├── requirements.txt          # Python dependencies
├── graph_rag.py              # Main pipeline execution script
├── generate_report.py        # Lab report compiler script
├── extracted_triples.json    # Cached LLM-extracted triples
├── benchmark_results.json    # 20 benchmark questions outputs
├── GraphRAG_Lab_Report.md    # Generated comparative analysis report
└── knowledge_graph.png       # Knowledge graph visualization (Matplotlib)
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- Access to an OpenAI-compatible LLM API endpoint

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration (`.env`)
Create a `.env` file in the root directory (or ensure the configured path in `graph_rag.py` exists) with your API details:
```env
OPENAI_BASE_URL="http://14.241.208.1:443/v1"
OPENAI_API_KEY="your-api-key-here"
MODEL_NAME="/var/lib/vllm/hf/gpt-oss-120b"
```

---

## Execution Guide

### 1. Run the RAG Pipeline
Run the main script to build the graph, visualize it, and evaluate both RAG systems:
```bash
python3 graph_rag.py
```

*Note: The first run will fetch triples via the LLM API and download the local embedding model (~90MB). Subsequent runs will instantly load the cached `extracted_triples.json`.*

#### Dry-Run (Quick Test)
To verify everything is working without running all 70 documents, you can run a dry-run using the first 2 documents and 1 test query:
```bash
python3 graph_rag.py --dry-run
```

### 2. Compile the Lab Report
Run the helper script to aggregate token usage, cost stats, benchmark answers, and compile the report:
```bash
python3 generate_report.py
```
This writes the final results directly to [GraphRAG_Lab_Report.md](GraphRAG_Lab_Report.md).
