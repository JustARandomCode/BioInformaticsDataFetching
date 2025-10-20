Bioinformatics Backend

#Overview
This project is a FastAPI-based backend for an AI-powered research paper analysis platform that integrates real biomedical data from PubMed (NCBI E-Utilities) with LLM-driven summarization and gene extraction using LangChain and Ollama (Llama2 model).
It automates the process of:
Searching research papers from PubMed based on user queries.
Summarizing retrieved papers using an LLM.
Analyzing selected papers to identify key genes and their biological roles.
Discovering gene-focused research trends across studies.

#Features
Feature	Description
Paper Search	Fetches relevant papers from PubMed using the NCBI E-Utilities API.
Smart Query Enhancement	Expands user queries with domain-specific filters (e.g., genetics, therapy).
Paper Summarization	Generates LLM-powered summaries highlighting key findings and research areas.
Gene Extraction	Identifies genes, symbols, organisms, and functions from abstracts.
Gene-Focused Research	Finds and summarizes papers centered around a given gene.
Async I/O	Uses aiohttp and asyncio for concurrent API calls to improve speed.
CORS Enabled	Supports cross-origin requests for integration with web UIs or dashboards.

#Tech Stack

Language: Python 3.9+
Framework: FastAPI
Async HTTP Client: aiohttp
Model Management: LangChain
LLM Backend: Ollama (llama2)
Data Source: PubMed (NCBI E-Utilities)
Middleware: CORS
Serialization: Pydantic Models

#Data Flow Summary
Step	        Input              	Output
1	            query	              List of papers from PubMed
2	            Paper data	        LLM-generated search summary
3	            Paper ID	          Identified genes + paper summary
4	            Gene name	          Gene-related papers + research trends

#Notes
The backend directly communicates with PubMed APIs using real data, not mock datasets.
Each API call is rate-limited (asyncio.sleep) to respect NCBI rate policies.
LLM output is parsed into JSON and mapped to defined Pydantic models.
The Ollama model (llama2) must be running locally or accessible via API.
