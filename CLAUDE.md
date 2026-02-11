# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChatMyCV is a production-grade RAG (Retrieval-Augmented Generation) chatbot that lets users interact with a CV/resume through an interview-style conversation. It uses Azure OpenAI for both LLM completions and embeddings, ChromaDB for vector storage, and includes comprehensive LLMOps infrastructure for observability, memory management, multi-agent routing, and human-in-the-loop review.

## Commands

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Run Flask Backend (port 8000)
```bash
python backend/app.py
```

### Run Streamlit UI (port 8501)
```bash
streamlit run streamlit_app.py
```

### Process CV Documents into Vector Store
```bash
# Via API after backend is running:
curl -X POST http://localhost:8000/process/process_file -H "Content-Type: application/json" -d '{"lang": "en"}'
```

### No test suite or linter is configured.

## Architecture

### Two Frontend Options
- **Static HTML** (`frontend/`): standalone HTML/CSS/JS that calls the Flask API
- **Streamlit** (`streamlit_app.py`): imports `ChatService` directly from backend (no HTTP needed)

### Backend (`backend/`)

**Entry point:** `app.py` creates a Flask app with two blueprint groups:
- `/chat` routes (`routes/chat_routes.py`) — chat, stream, clear session
- `/process` routes (`routes/doc_process_routes.py`) — ingest documents, delete collections

**Service layer** (`services/`):
- `chat_serv.py` — `ChatService` orchestrates the RAG pipeline: compose retrieval query from history + user query, get embedding, query ChromaDB, build prompt with context, call LLM, parse `<answer>` tags from response
- `doc_processor_serv.py` — `DocProcessor` reads markdown files from `data/{lang}/`, generates embeddings, stores in ChromaDB
- `prompter.py` — System prompts for two interviewer personas (HR and Engineer), both use chain-of-thought with `<thinking>`/`<answer>` XML tags

**Singleton initialization pattern:** `services/__init__.py` instantiates `chat_service = ChatService()` at import time. Similarly, `modules/__init__.py` creates `azure_client = AzureModule()`. These singletons are used throughout.

**LLM module** (`modules/`):
- `base.py` — abstract `BaseLLMProvider` interface
- `azure_module.py` — Azure OpenAI implementation (chat completion, streaming, embeddings). Reads credentials from `backend/.env`.

**Vector store** (`vectorstores/chroma_vectordb.py`): wraps ChromaDB with persistent storage under `.chroma/`. Two collections: `chat_cv_en` and `chat_cv_zhtw`.

**Parsers** (`parsers/`): `markdown_parser.py` is the primary parser (splits by headers into `Node` objects). PDF/DOCX/TXT parsers are stubs.

### Data Flow (Chat Request)
1. User query arrives with `lang`, `query`, `session_id`, `character` (hr/engineer)
2. `ChatService.chat()` retrieves conversation history from in-memory `_ConversationStore`
3. Composes retrieval query (user query + recent history)
4. Gets embedding via `AzureModule.get_embedding()`, queries ChromaDB for top-k similar docs
5. Builds message list: system prompt (persona-specific) + few-shot CoT example + context + history + user query
6. Calls Azure OpenAI, parses final answer from `<answer>` XML tags
7. Persists conversation in `_ConversationStore` (in-memory, 5-min idle expiry)

### Session Management
- In-memory only (`_ConversationStore` in `chat_serv.py`) with `threading.RLock`
- Sessions auto-expire after 300 seconds of inactivity
- Session IDs are UUIDs

### Environment Variables (backend/.env)
Required Azure OpenAI credentials: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_BASE`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_LLM_ENGINE`, `AZURE_OPENAI_LLM_MODEL`, `AZURE_OPENAI_EMBED_ENGINE`, `AZURE_OPENAI_EMBED_MODEL`, `EMBED_DIM`, `EMBED_TIMEOUT`.

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/healthz` | Health check |
| POST | `/chat/` | Chat with RAG |
| POST | `/chat/stream` | SSE streaming chat |
| POST | `/chat/clear` | Clear one session |
| POST | `/chat/clear_all` | Clear all sessions |
| POST | `/process/process_file` | Ingest CV markdown files |
| DELETE | `/process/collection` | Delete a vector collection |

## Key Conventions

- CV data files go in `backend/data/en/` (English) or `backend/data/zhtw/` (Traditional Chinese) as markdown
- LLM responses use XML-style `<thinking>` and `<answer>` tags; only content inside `<answer>` is returned to users
- The `character` parameter controls which system prompt is used: `"hr"` for non-technical HR style, `"engineer"` for technical bullet-point style
- Python path manipulation (`sys.path.append`) is used in several files to handle imports between `backend/` and project root

## LLMOps Architecture

The system includes production-grade LLMOps features for observability, memory persistence, intelligent routing, and quality control.

### 1. Langfuse Observability

**Purpose**: Full tracing of LLM operations for debugging, performance monitoring, and cost tracking.

**Configuration** (`backend/.env`):
```
LANGFUSE_PUBLIC_KEY=pk-lf-local
LANGFUSE_SECRET_KEY=sk-lf-local
LANGFUSE_HOST=http://localhost:3000
```

**Implementation** (`backend/observability/langfuse_client.py`):
- Singleton Langfuse client with graceful degradation
- Automatic trace creation for each chat request
- Spans for each pipeline stage:
  - `memory-load` - Conversation history retrieval
  - `router-decision` - Agent routing logic
  - `vector-retrieval` - ChromaDB query with similarity scores
  - `prompt-construction` - Message building
  - `llm-call` - Azure OpenAI completion with usage stats
  - `hitl-trigger` - HITL decision gate (if triggered)

**Usage**:
- Every chat request gets a unique `trace_id` returned in the response
- View traces in Langfuse UI at `http://localhost:3000`
- Traces include full metadata: query, retrieved docs, similarity scores, LLM usage

### 2. Redis Memory System

**Purpose**: Persistent session-based conversational memory (replaces in-memory store).

**Configuration** (`backend/.env`):
```
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

**Implementation** (`backend/services/memory_serv.py`):
- `RedisMemoryStore` class with Redis lists for conversation history
- Automatic TTL (24 hours) for session cleanup
- Configurable max messages per session (default 20)
- Graceful fallback to in-memory if Redis unavailable

**Key Methods**:
- `save_message(session_id, role, content)` - Append user/assistant messages
- `load_history(session_id, max_messages)` - Retrieve recent conversation
- `clear_session(session_id)` - Delete session
- `get_last_session()` - Find most recent session

**Integration**:
- `ChatService` automatically uses Redis if available
- Maintains backward compatibility with in-memory `_ConversationStore`

### 3. Multi-Agent Router

**Purpose**: Intelligent query routing to specialized agents based on intent detection.

**Agents**:
1. **RAG Agent** (default) - Document-based queries about CV content
   - Triggers: CV keywords (experience, skills, education, projects)
   - Uses full RAG pipeline: retrieval → context → LLM

2. **Chat Agent** - Greetings and casual conversation
   - Triggers: hi, hello, thanks, goodbye (bilingual)
   - Returns predefined friendly responses without LLM call

3. **Memory Agent** - Questions about previous conversation
   - Triggers: "earlier", "previously", "what did I ask", "剛剛", "之前"
   - Retrieves and formats conversation history

**Implementation** (`backend/services/router.py`):
- Rule-based routing with bilingual keyword matching
- Language-aware triggers for English and Traditional Chinese
- Returns route decision: `"rag"`, `"chat"`, or `"memory"`

**Flow**:
```
User Query → Router → [RAG Agent | Chat Agent | Memory Agent] → Response
```

**Response Fields**:
- `route`: Which agent handled the query
- Chat response includes route information for transparency

### 4. HITL (Human-In-The-Loop) Decision Gate

**Purpose**: Quality control for low-confidence or sensitive queries requiring human review.

**Trigger Conditions**:
1. **Low Confidence**: Average similarity score < 0.5
2. **Risky Keywords**: Sensitive topics (legal, financial, confidential)
   - English: claim, policy, legal, liability, contract
   - Chinese: 理賠, 條款, 法律, 責任, 機密
3. **Uncertainty**: LLM response contains uncertainty indicators

**Configuration** (`backend/.env`):
```
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=chatmycv
POSTGRES_USER=chatmycv_user
POSTGRES_PASSWORD=chatmycv_pass
```

**Database Schema** (`backend/db/schema.sql`):
- `hitl_reviews` table with columns:
  - `id`, `session_id`, `question`, `retrieved_docs` (JSONB)
  - `similarity_score`, `reason`, `status` (pending/approved/rejected)
  - `reviewer`, `final_answer`, `created_at`, `reviewed_at`

**Implementation** (`backend/services/hitl_serv.py`):
- `HITLService` class for decision logic and database operations
- PostgreSQL audit trail for all HITL requests
- Review workflow: pending → approved/rejected

**API Endpoints** (`/hitl/*`):
- `GET /hitl/pending` - List pending reviews
- `GET /hitl/{id}` - Get review details
- `POST /hitl/{id}/approve` - Approve with human answer
- `POST /hitl/{id}/reject` - Reject review
- `GET /hitl/stats` - System statistics

**User Experience**:
When HITL triggers, user receives:
```
"Thank you for your question. This query requires human review for accuracy.
Your request has been logged (Review ID: 123).
A human expert will review and provide an answer soon."
```

### 5. Prompt Evaluation Pipeline

**Purpose**: Regression testing for prompt optimization and quality assurance.

**Components**:
- `backend/evaluation/eval_dataset.json` - Test questions with expected keywords
- `backend/evaluation/evaluator.py` - Keyword matching evaluation logic
- `backend/evaluation/run_eval.py` - CLI for running evaluations

**CLI Usage**:
```bash
# Run English evaluation with HR persona
python backend/evaluation/run_eval.py --lang en --character hr

# Run Chinese evaluation with Engineer persona
python backend/evaluation/run_eval.py --lang zhtw --character engineer --k 3

# Custom output path
python backend/evaluation/run_eval.py --lang en --output results/eval_20260212.json
```

**Metrics**:
- **Score**: Percentage of expected keywords matched in answer
- **Category Breakdown**: Scores by question type (experience, skills, education)
- **Pass Threshold**: 60% keyword match
- **Performance**: Average response time

**Output**:
```
EVALUATION SUMMARY
===============================================================
Total Questions:    10
Average Score:      78.50%
Average Time:       2.34s
Result:             PASSED ✓

Category Breakdown:
  experience      85.00%
  skills          72.00%
  education       80.00%
```

### 6. Configuration Management

**Centralized Config** (`backend/config.py`):
- Type-safe configuration classes with dataclasses
- Validation for required vs optional components
- Environment variable loading with defaults
- Configuration summary printing

**Usage**:
```python
from backend.config import config

# Check if features are enabled
if config.langfuse.enabled:
    # Use Langfuse tracing

if config.postgres.enabled:
    # Use HITL features
```

## Updated Data Flow (with LLMOps)

1. **Request Arrives** → Create Langfuse trace with `trace_id`
2. **Router Decision** → Determine agent (RAG/Chat/Memory) based on query analysis
3. **Memory Load** (Span) → Retrieve conversation history from Redis
4. **Agent Execution**:
   - **Chat Agent**: Return predefined response
   - **Memory Agent**: Format and return conversation history
   - **RAG Agent**: Continue to steps 5-9
5. **Vector Retrieval** (Span) → Query ChromaDB, calculate similarity scores
6. **HITL Check** → If low confidence/risky, save to PostgreSQL and return HITL message
7. **Prompt Construction** (Span) → Build messages with context and history
8. **LLM Call** (Span) → Azure OpenAI completion, track usage
9. **Response** → Parse `<answer>` tags, save to Redis, return with metadata

**Response includes**:
- `response`: The answer text
- `trace_id`: Langfuse trace ID for debugging
- `route`: Which agent handled the query
- `hitl_triggered`: Boolean indicating if HITL was triggered
- `avg_similarity`: Average retrieval confidence score
- `retrieved_docs_count`: Number of context documents used

## Infrastructure Setup

### Docker Compose (Quick Start)

Start all LLMOps services with one command:
```bash
docker-compose up -d
```

**Services**:
- **Langfuse** (port 3000) - Observability UI + API
- **Langfuse PostgreSQL** (port 5432) - Langfuse data
- **Langfuse ClickHouse** (ports 8123, 9000) - Analytics
- **Redis** (port 6379) - Session memory
- **PostgreSQL** (port 5433) - HITL audit trail

**Langfuse Access**:
- URL: `http://localhost:3000`
- Email: `admin@chatmycv.local`
- Password: `admin123`

### Manual Setup

**Redis**:
```bash
# Windows (via Chocolatey)
choco install redis-64

# Start service
redis-server
```

**PostgreSQL**:
```bash
# Windows (via Chocolatey)
choco install postgresql

# Create database and user
psql -U postgres
CREATE DATABASE chatmycv;
CREATE USER chatmycv_user WITH PASSWORD 'chatmycv_pass';
GRANT ALL PRIVILEGES ON DATABASE chatmycv TO chatmycv_user;

# Run schema
psql -U chatmycv_user -d chatmycv -f backend/db/schema.sql
```

**Langfuse** (Self-Hosted):
See `docker-compose.yml` or visit https://langfuse.com/docs/deployment/self-host

## Updated API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/healthz` | Health check |
| POST | `/chat/` | Chat with RAG (returns trace_id, route, hitl info) |
| POST | `/chat/stream` | SSE streaming chat |
| POST | `/chat/clear` | Clear one session |
| POST | `/chat/clear_all` | Clear all sessions |
| POST | `/process/process_file` | Ingest CV markdown files |
| DELETE | `/process/collection` | Delete a vector collection |
| **GET** | **`/hitl/pending`** | **List pending HITL reviews** |
| **POST** | **`/hitl/{id}/approve`** | **Approve HITL review** |
| **POST** | **`/hitl/{id}/reject`** | **Reject HITL review** |
| **GET** | **`/hitl/stats`** | **HITL statistics** |
| **GET** | **`/observability/health`** | **Component health status** |
| **GET** | **`/observability/stats`** | **System statistics** |
| **POST** | **`/observability/flush`** | **Flush Langfuse traces** |

## Updated Session Management

- **Persistent**: Redis-backed with 24-hour TTL (configurable)
- **Fallback**: In-memory if Redis unavailable
- **Cleanup**: Automatic via Redis TTL (no manual cleanup needed)
- **Session IDs**: UUIDs
- **History Limit**: 20 messages per session (configurable)

## Updated Environment Variables

See `backend/.env` for full configuration. Key additions:

**Langfuse**:
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

**Redis**:
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`

**PostgreSQL** (HITL):
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
