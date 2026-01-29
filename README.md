# ChatMyCV

**Interactive AI Resume Powered by RAG**

A conversational interface that enables recruiters and hiring managers to query my professional experience in natural language. Built with multi-LLM support and ChromaDB, featuring persona-based responses and bilingual support.

## Features

- **Multi-LLM Provider Support**: Choose from Azure OpenAI, OpenAI, Claude (Anthropic), or Gemini (Google)
- **Flexible Embedding**: Independent embedding provider selection (Azure OpenAI, OpenAI, or Gemini)
- **Natural Language Querying**: Ask questions about work experience, skills, and background in conversational English or Traditional Chinese
- **RAG Pipeline**: Retrieval-Augmented Generation ensures responses are grounded in actual CV content
- **Dual Interview Personas**:
  - **HR Representative**: Conversational, high-level overview with warm tone
  - **Engineering Manager**: Technical, precise responses with bullet-point format
- **Bilingual Support**: Full support for English and Traditional Chinese with separate vector collections
- **Streaming Responses**: Real-time token-by-token response streaming
- **Session Management**: Conversation history with automatic cleanup

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│   Flask API     │────▶│   LLM Provider  │
│   (Frontend)    │◀────│   (Backend)     │◀────│ (Chat/Streaming)│
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                        ┌────────┴────────┐
                        ▼                 ▼
               ┌─────────────────┐ ┌─────────────────┐
               │    ChromaDB     │ │ Embed Provider  │
               │  (Vector Store) │ │  (Embeddings)   │
               └─────────────────┘ └─────────────────┘
```

### Data Flow

```
User Query → Embed Query → Retrieve Top-K Documents → Format Context → LLM Generation → Stream Response
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Backend API | Flask |
| LLM Providers | Azure OpenAI, OpenAI, Claude, Gemini |
| Embedding Providers | Azure OpenAI, OpenAI, Gemini |
| Vector DB | ChromaDB |
| Document Processing | Custom Markdown Parser |

## Project Structure

```
ChatMyCV/
├── streamlit_app.py        # Main Streamlit application
├── backend/
│   ├── app.py              # Flask REST API
│   ├── config/
│   │   └── prompts/        # Persona-specific system prompts
│   ├── db/
│   │   └── chroma_vectordb.py  # ChromaDB wrapper
│   ├── llm/
│   │   ├── base.py             # Abstract LLM base class
│   │   ├── azure_module.py     # Azure OpenAI integration
│   │   ├── openai_module.py    # OpenAI integration
│   │   ├── claude_module.py    # Claude (Anthropic) integration
│   │   └── gemini_module.py    # Gemini (Google) integration
│   ├── services/
│   │   ├── chat_serv.py        # Chat service with RAG
│   │   └── doc_processor_serv.py  # Document indexing
│   ├── parsers/
│   │   └── markdown_parser.py  # Markdown chunking
│   └── data/
│       ├── en/             # English CV documents
│       └── zhtw/           # Traditional Chinese CV documents
├── frontend/               # Alternative HTML/JS frontend
├── utils/
│   └── app_logger.py       # Logging utilities
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.11+
- API access to at least one LLM provider:
  - Azure OpenAI
  - OpenAI
  - Anthropic (Claude)
  - Google (Gemini)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ChatMyCV.git
cd ChatMyCV
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp backend/.env.example backend/.env
# Edit .env with your API credentials
```

Required environment variables:
```bash
# Provider Selection
LLM_PROVIDER=azure          # Options: azure, openai, claude, gemini
EMBED_PROVIDER=azure        # Options: azure, openai, gemini (claude not supported)

# Azure OpenAI (if using azure)
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_BASE=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_LLM_ENGINE=gpt-4o
AZURE_OPENAI_EMBED_ENGINE=text-embedding-3-small

# OpenAI (if using openai)
OPENAI_API_KEY=your_api_key
OPENAI_LLM_MODEL=gpt-4o
OPENAI_EMBED_MODEL=text-embedding-3-small

# Claude (if using claude)
ANTHROPIC_API_KEY=your_api_key
CLAUDE_MODEL=claude-sonnet-4-20250514

# Gemini (if using gemini)
GOOGLE_API_KEY=your_api_key
GEMINI_LLM_MODEL=gemini-2.0-flash
GEMINI_EMBED_MODEL=text-embedding-004
```

### Running the Application

**Option 1: Streamlit (Recommended)**
```bash
streamlit run streamlit_app.py
```

**Option 2: Flask API**
```bash
python backend/app.py
```

### Indexing Your CV

Place your CV/resume in markdown format under:
- `backend/data/en/` for English
- `backend/data/zhtw/` for Traditional Chinese

Then process and index:
```bash
python backend/services/doc_processor_serv.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat/` | Chat with RAG (non-streaming) |
| POST | `/chat/stream` | Chat with streaming response |
| POST | `/chat/clear` | Clear session history |
| POST | `/process/process_file` | Index a document |
| DELETE | `/process/collection` | Delete vector collection |
| GET | `/healthz` | Health check |

## Evaluation Approach

The system is designed with factual accuracy in mind:

- **Context Grounding**: All responses are generated from retrieved CV sections
- **Source Attribution**: Retrieved documents include metadata for citation
- **Persona Consistency**: System prompts ensure appropriate tone and detail level
- **Fallback Handling**: Graceful responses when information is not available in the CV


## Evaluation Framework

A comprehensive evaluation system to ensure factual accuracy and proper citation:

### Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| **LLM-as-Judge** | Faithfulness | Is the answer grounded in retrieved context? |
| | Relevance | Does the answer address the question? |
| | Citation | Are sources properly attributed? |
| **Retrieval** | Hit Rate | % of expected sources retrieved |
| | MRR | Mean Reciprocal Rank of relevant docs |
| | P@K | Precision at K retrieved documents |

### Running Evaluation

```bash
# Run evaluation on English test cases
python backend/evaluation/run_evaluation.py --language en

# Run on Traditional Chinese
python backend/evaluation/run_evaluation.py --language zhtw

# Custom test cases
python backend/evaluation/run_evaluation.py -t path/to/test_cases.json
```

### Test Case Format

```json
{
  "test_cases": [
    {
      "id": "tc_001",
      "question": "What is the candidate's current role?",
      "expected_sources": ["Work Experience"],
      "ground_truth": "Description of expected answer",
      "category": "factual"
    }
  ]
}
```

### Optional: RAGAS Integration

For advanced retrieval metrics, install RAGAS:
```bash
pip install ragas datasets langchain-openai
```

## License

MIT License

## Author

Vivi Liu
