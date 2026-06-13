# Bioinformatics Backend

An AI-powered FastAPI backend for biomedical research paper analysis that combines real-time PubMed data retrieval with Large Language Model (LLM) intelligence for summarization, gene extraction, and research trend discovery.

The platform integrates PubMed (NCBI E-Utilities), LangChain, and Ollama (Llama2) to automate literature analysis, helping researchers identify relevant studies, extract biological insights, and explore gene-focused research trends.

---

## Overview

This backend automates the process of:

* Searching biomedical research papers from PubMed
* Summarizing retrieved papers using an LLM
* Extracting genes and their biological functions from research abstracts
* Discovering gene-focused research trends across studies
* Delivering structured results through REST APIs

---

## Features

### Paper Search

Fetches relevant biomedical research papers directly from PubMed using the NCBI E-Utilities API.

### Smart Query Enhancement

Improves search relevance by automatically expanding user queries with domain-specific biomedical keywords and filters.

### Paper Summarization

Generates concise summaries highlighting:

* Key findings
* Research objectives
* Experimental outcomes
* Biological significance

### Gene Extraction

Identifies and extracts:

* Gene names
* Gene symbols
* Associated organisms
* Biological functions
* Molecular pathways

### Gene-Focused Research

Searches and summarizes studies related to a specific gene and identifies emerging research trends.

### Asynchronous Processing

Uses `aiohttp` and `asyncio` for concurrent API calls and improved performance.

### CORS Support

Enables seamless integration with frontend applications and dashboards.

---

## Technology Stack

| Component     | Technology                |
| ------------- | ------------------------- |
| Language      | Python 3.9+               |
| Framework     | FastAPI                   |
| LLM Framework | LangChain                 |
| LLM Backend   | Ollama (Llama2)           |
| Data Source   | PubMed (NCBI E-Utilities) |
| HTTP Client   | aiohttp                   |
| Validation    | Pydantic                  |
| Concurrency   | asyncio                   |
| Middleware    | CORS                      |

---

## Architecture

```text
User Query
    │
    ▼
PubMed Search
    │
    ▼
Paper Retrieval
    │
    ▼
LLM Processing (LangChain + Ollama)
    │
    ├── Paper Summarization
    │
    └── Gene Extraction
            │
            ▼
     Gene Trend Analysis
            │
            ▼
        JSON Response
```

---

## Data Flow

| Step | Input          | Output                         |
| ---- | -------------- | ------------------------------ |
| 1    | Search Query   | List of PubMed Papers          |
| 2    | Paper Metadata | AI-Generated Summary           |
| 3    | Paper ID       | Genes and Biological Insights  |
| 4    | Gene Name      | Gene-Related Papers and Trends |

---

## Project Structure

```text
bioinformatics-backend/
│
├── app/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   ├── models/
│   ├── schemas/
│   └── utils/
│
├── requirements.txt
├── README.md
└── .env
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/bioinformatics-backend.git
cd bioinformatics-backend
```

### Create a Virtual Environment

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Ollama Setup

Install Ollama:

```text
https://ollama.com/download
```

Pull the Llama2 model:

```bash
ollama pull llama2
```

Start Ollama:

```bash
ollama serve
```

Verify installation:

```bash
ollama run llama2
```

---

## Running the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The application will be available at:

```text
http://localhost:8000
```

### API Documentation

Swagger UI:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

---

## Example API Endpoints

### Search Papers

**POST** `/search`

Request:

```json
{
  "query": "breast cancer gene therapy"
}
```

Response:

```json
{
  "papers": [
    {
      "pmid": "123456",
      "title": "Gene Therapy in Breast Cancer"
    }
  ]
}
```

---

### Summarize Papers

**POST** `/summarize`

Request:

```json
{
  "query": "breast cancer gene therapy"
}
```

Response:

```json
{
  "summary": "Recent studies focus on..."
}
```

---

### Analyze Paper

**POST** `/analyze-paper`

Request:

```json
{
  "pmid": "123456"
}
```

Response:

```json
{
  "genes": [
    {
      "symbol": "BRCA1",
      "function": "DNA repair"
    }
  ]
}
```

---

### Gene Analysis

**POST** `/gene-analysis`

Request:

```json
{
  "gene": "TP53"
}
```

Response:

```json
{
  "gene": "TP53",
  "research_trends": [
    "Cancer progression",
    "Tumor suppression",
    "Precision medicine"
  ]
}
```

---

## Rate Limiting

To comply with NCBI usage guidelines:

* Requests are throttled using `asyncio.sleep()`
* API calls are executed asynchronously
* PubMed rate limits are respected

For production deployments, consider:

* Redis caching
* Request queuing
* Exponential backoff
* Persistent storage

---

## Notes

* Uses real-time PubMed data rather than mock datasets.
* LLM outputs are converted into structured JSON responses.
* All responses are validated using Pydantic models.
* Ollama must be running before starting the application.
* Designed for integration with research tools and analytics platforms.

---

## Use Cases

* Biomedical Literature Review
* Gene Discovery Research
* Drug Target Identification
* Scientific Knowledge Mining
* Academic Research Assistance
* Bioinformatics Pipelines
* Research Trend Analysis

---

## Future Enhancements

* Vector Database Integration
* Semantic Search with Embeddings
* Multi-Model Support
* Citation Network Analysis
* Knowledge Graph Generation
* PDF Upload and Analysis
* Retrieval-Augmented Generation (RAG)
* Research Paper Recommendation Engine

---

FastAPI, LangChain, Ollama, and PubMed APIs to accelerate biomedical research and scientific literature analysis.
