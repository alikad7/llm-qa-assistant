
# Software Testing Knowledge Assistant

A domain-specific Retrieval-Augmented Generation (RAG) application designed to answer questions about software testing using pre-built retrieval indexes, hybrid search, reranking, query analysis, and LLM-based answer generation.

This project was developed as an educational RAG system focused on **Software Testing Knowledge**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Main Features](#main-features)
- [Main Pipeline](#main-pipeline)
- [Models Folder Notice](#models-folder-notice)
- [Environment Variables](#environment-variables)
- [Installation](#installation)
- [How to Run the Application](#how-to-run-the-application)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Educational Purpose](#educational-purpose)

---

## Project Overview

Software Testing Knowledge Assistant is a Streamlit-based RAG application that helps users ask questions about software testing topics.

The system retrieves relevant information from pre-built FAISS and BM25 indexes and uses the retrieved context to generate grounded answers with an LLM.

The knowledge base was created from selected software testing resources. The original PDF source files are not included in this repository due to copyright considerations. Instead, the generated indexes are included so the application can be used without rebuilding the knowledge base.

---

## Main Features
<img width="1536" height="1024" alt="pipeline" src="https://github.com/user-attachments/assets/abb263d3-0db3-47db-8574-00811016b58c" />


- Domain-specific question answering for software testing
- Streamlit-based user interface
- Pre-built FAISS vector index
- Pre-built BM25 keyword index
- Hybrid retrieval using semantic and lexical search
- LLM-based query analysis
- Rule-based fallback query analysis
- Query enhancement for improved retrieval
- Software testing domain classification
- Intent detection
- Retrieval-specific query generation
- Cross-encoder reranking
- MMR-based post-processing for result diversification
- Context construction for LLM prompting
- LLM-based answer generation
- Retrieval evaluation support
- Generation evaluation support
- Local model loading support
- Environment-based API configuration

---

## Main Pipeline

The application follows a modular Retrieval-Augmented Generation pipeline.

1. **User Question Input**: The user enters a software testing question through the Streamlit interface. (e.g., "What is exploratory testing?")

2. **Query Analysis and Query Enhancement**: Before retrieval, the question is analyzed by the `LLMQueryAnalyzer` or `FallbackQueryAnalyzer`. Query enhancement includes:
    - Domain classification
    - Intent detection
    - Query normalization
    - Software testing keyword recognition
    - Semantic query generation (for FAISS)
    - Keyword-oriented query generation (for BM25)
    - Guardrails for out-of-domain questions

3. **Hybrid Retrieval**: The system retrieves candidate chunks using:
    - **FAISS Vector Search**: Semantic similarity search.
    - **BM25 Keyword Search**: Lexical search for exact or highly relevant keywords.

4. **Result Fusion and Scoring**: Results are combined into a single candidate list with scoring logic to penalize low-quality/short chunks.

5. **Cross-Encoder Reranking**: A local cross-encoder model re-scores candidate chunks for direct relevance.

6. **MMR Post-Processing**: Maximal Marginal Relevance is used to diversify results and avoid redundant information.

7. **Context Building**: Selected chunks are combined into a final prompt context for the LLM.

8. **Answer Generation**: The LLM generates the final answer based on the retrieved context.

---

## Models Folder Notice

The `models/` folder is required for local model loading. Expected structure:

```text
models/
|-- ms-marco-MiniLM-L-6-v2/
`-- cross-encoder/
    `-- ms-marco-MiniLM-L-6-v2/
```

- **Important:** Do not rename model folders unless you update `src/config.py`.
- **Note:** If the embedding model changes, the FAISS index must be rebuilt.
- Model download link: `(https://drive.google.com/file/d/1Hlrcrl9uEXM0zMdFdFfOo1ZyygAlgW2g/view?usp=drive_link)`

---

## Environment Variables

Create a `.env` file in the project root:

```env
GAPGPT_API_KEY=your_api_key_here
GAPGPT_BASE_URL=https://api.gapgpt.app/v1
CHAT_MODEL=gpt-4o-mini
```

---

## Installation

1. **Clone the repository**
```bash
git clone <REPOSITORY_URL>
cd <PROJECT_DIRECTORY>
```

2. **Create a virtual environment (Windows PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. **Install dependencies**
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Run the Application

Run the Streamlit app:
```bash
streamlit run app.py
```
If it does not open automatically, visit `http://localhost:8501`.

**Important:** For normal usage, you do not need to run `python rebuild_indexes.py`. The indexes are already provided. Only rebuild if you change documents, embedding models, or chunking logic.

---

## Evaluation

Evaluation components are located under `src/evaluation/`.

- **Retrieval Evaluation**: Measures Precision@K, Recall@K, MRR.
- **Generation Evaluation**: Measures relevance, faithfulness, and context usage.

To run generation evaluation:
```bash
python run_generation_evaluation.py
```

---

## Project Structure

```text
.
├── .env
├── .env.example
├── .gitignore
├── README.md
├── app.py
├── rebuild_indexes.py
├── requirements.txt
├── run_generation_evaluation.py
├── data/
│   ├── urls.txt
│   └── indexes/
│       ├── bm25/
│       └── faiss_index/
├── models/
│   ├── ms-marco-MiniLM-L-6-v2/
│   └── cross-encoder/
└── src/
    ├── ingestion/
    ├── processing/
    ├── query_analyzer/
    ├── retrieval/
    ├── generation/
    └── evaluation/
```

---

## Limitations

- The system is specialized for software testing and may reject unrelated questions.
- The system supports English-language queries only and does not process inputs in other languages
- Answer quality depends on the knowledge base coverage.
- Original PDF documents are not included due to copyright.
- FAISS indexes are tied to the specific embedding model used during creation.
- Reranking and local model loading may increase response time and resource requirements.

---

## Educational Purpose

This project was developed for educational purposes to demonstrate a practical implementation of a RAG system, including document preprocessing, hybrid retrieval, query enhancement, reranking, and evaluation.
