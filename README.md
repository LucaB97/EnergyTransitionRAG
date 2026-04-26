# Introduction

This system helps environmental researchers and policymakers answer questions about the social impacts of renewable energy adoption by synthesizing evidence from multiple peer-reviewed research papers. 
Given a query (e.g., “How are local communities affected by large renewable projects?”), the system retrieves relevant passages from a corpus of open-access climate research, aggregates key findings, and produces a structured synthesis with explicit citations. 
The goal is to support evidence-based decision-making while minimizing hallucinations and maximizing transparency.

# Workflow

<figure align="center">

<span style="display: inline-block; text-align: center; width: 70%; height: 70%;">
<img src="img/workflow.png"/>
</span>

</figure>

## Getting Started:
### Prerequisites
- Python 3.8.17+
- Conda (strongly recommended, especially on Windows)

### Environment setup
Create and activate a conda environment:
```bash
conda create -n env-research-synth python=3.8.17
conda activate env-research-synth
```

### Install dependencies
- Install FAISS using conda (recommended to avoid compatibility issues):
```bash
conda install -c conda-forge faiss-cpu
```
- Install remaining dependencies: 
```bash
pip install -r requirements.txt
```

### Project setup
- Place PDFs in `data/papers/`
- Ensure `metadata.csv` is present in `data/`

## Project Phases
### Phase 0: Data Collection
- Collect open-access research papers in PDF format on the topic.
- Create a `metadata.csv` file summarizing each paper: ID, title, year, publishing journal.

### Phase 1: Document Processing & Extraction, Chunking
`python -m scripts.extraction --chunk-size 500 --overlap 100`
- Convert PDFs to text.
- Remove common headers, footers, and references.
- Normalize whitespace.
- Split text into chunks.
- Save chunks in JSON format with metadata: chunk ID, paper ID, title, year.

### Phase 2: Embeddings & Indexing
`python -m scripts.indexing --chunk-size 500 --overlap 100 --embedding openai`

### Phase 3: Parameters tuning
`python -m scripts.tuning --chunk-size 500 --overlap 100 --embedding openai` 

### Phase 3: Semantic Retrieval & LLM-Based Summarization (RAG)
`python -m scripts.test___retrieval_and_synthesis`
• Perform semantic search on user queries to retrieve relevant passages.
• Use a language model to synthesize retrieved passages into structured answers with citations.

### Phase 4: API & Minimal Deployment
- Expose the system via a simple API for querying and returning structured responses.

### Phase 5: Documentation & Reflection
- Reflect on the system design, challenges, and potential improvements.
