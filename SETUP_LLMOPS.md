# ChatMyCV LLMOps Setup Guide

This guide will help you set up the complete LLMOps infrastructure for ChatMyCV, including Langfuse observability, Redis memory, PostgreSQL HITL, and all related services.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (Docker Compose)](#quick-start-docker-compose)
3. [Manual Setup](#manual-setup)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Production Considerations](#production-considerations)

---

## Prerequisites

### Required Software

- **Python 3.9+** - For running the backend
- **Docker & Docker Compose** - For LLMOps services (recommended)
- OR manually install:
  - **Redis 7+** - Session memory
  - **PostgreSQL 15+** - HITL audit trail

### System Requirements

- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 4GB minimum (8GB recommended for Docker stack)
- **Disk**: 5GB free space
- **Network**: Internet access for initial setup

---

## Quick Start (Docker Compose)

The fastest way to get all LLMOps services running is using Docker Compose.

### Step 1: Start Services

From the project root directory:

```bash
docker-compose up -d
```

This will start:
- **Langfuse** UI (http://localhost:3000)
- **Langfuse PostgreSQL** (port 5432)
- **Langfuse ClickHouse** (ports 8123, 9000)
- **Redis** (port 6379)
- **PostgreSQL HITL** (port 5433)

### Step 2: Verify Services

Check that all containers are running:

```bash
docker-compose ps
```

Expected output:
```
NAME                        STATUS         PORTS
chatmycv-langfuse           Up            0.0.0.0:3000->3000/tcp
chatmycv-langfuse-db        Up            0.0.0.0:5432->5432/tcp
chatmycv-langfuse-clickhouse Up           0.0.0.0:8123->8123/tcp, 0.0.0.0:9000->9000/tcp
chatmycv-redis              Up            0.0.0.0:6379->6379/tcp
chatmycv-hitl-db            Up            0.0.0.0:5433->5432/tcp
```

### Step 3: Access Langfuse UI

1. Open browser to http://localhost:3000
2. Login with default credentials:
   - **Email**: `admin@chatmycv.local`
   - **Password**: `admin123`
3. You should see the Langfuse dashboard

### Step 4: Configure Environment

Ensure your `backend/.env` file contains:

```env
# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-local
LANGFUSE_SECRET_KEY=sk-lf-local
LANGFUSE_HOST=http://localhost:3000

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# PostgreSQL (HITL)
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=chatmycv
POSTGRES_USER=chatmycv_user
POSTGRES_PASSWORD=chatmycv_pass
```

### Step 5: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `langfuse>=2.0.0`
- `redis>=5.0.0`
- `psycopg2-binary>=2.9.0`

### Step 6: Start Backend

```bash
python backend/app.py
```

Expected output:
```
INFO:ChatService:Using Redis memory store (connected: True)
INFO:LangfuseClient:Langfuse initialized successfully. Host: http://localhost:3000
INFO:HITLService:HITL service initialized successfully. Database: chatmycv
 * Running on http://0.0.0.0:8000
```

### Step 7: Test the System

Send a test chat request:

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "lang": "en",
    "query": "What is the candidate'\''s experience?",
    "character": "hr"
  }'
```

Expected response includes:
```json
{
  "status": "success",
  "response": "...",
  "trace_id": "langfuse-trace-id-here",
  "route": "rag",
  "hitl_triggered": false,
  "avg_similarity": 0.85
}
```

Check the Langfuse UI to see the trace!

---

## Manual Setup

If you prefer not to use Docker, you can install each component manually.

### Redis Setup

#### Windows (via Chocolatey)

```powershell
choco install redis-64
redis-server
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### macOS (via Homebrew)

```bash
brew install redis
brew services start redis
```

#### Verify Redis

```bash
redis-cli ping
# Expected: PONG
```

### PostgreSQL Setup

#### Windows (via Chocolatey)

```powershell
choco install postgresql
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### macOS (via Homebrew)

```bash
brew install postgresql
brew services start postgresql
```

#### Create HITL Database

```bash
# Connect as postgres user
psql -U postgres

# Create database and user
CREATE DATABASE chatmycv;
CREATE USER chatmycv_user WITH PASSWORD 'chatmycv_pass';
GRANT ALL PRIVILEGES ON DATABASE chatmycv TO chatmycv_user;
\q

# Run schema
psql -U chatmycv_user -d chatmycv -f backend/db/schema.sql
```

#### Verify PostgreSQL

```bash
psql -U chatmycv_user -d chatmycv -c "SELECT COUNT(*) FROM hitl_reviews;"
# Expected: count | 0
```

### Langfuse Setup (Self-Hosted)

For self-hosted Langfuse, we recommend using Docker. See the official docs:

https://langfuse.com/docs/deployment/self-host

Alternatively, use **Langfuse Cloud** (free tier available):

1. Sign up at https://langfuse.com
2. Create a project
3. Get your API keys from Settings
4. Update `backend/.env`:
   ```env
   LANGFUSE_PUBLIC_KEY=pk-lf-xxx
   LANGFUSE_SECRET_KEY=sk-lf-xxx
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

---

## Configuration

### Environment Variables

All configuration is in `backend/.env`:

```env
# Azure OpenAI (Required)
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_BASE=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_LLM_ENGINE=your-deployment-name
AZURE_OPENAI_LLM_MODEL=gpt-4.1-mini
AZURE_OPENAI_EMBED_ENGINE=your-embedding-deployment
AZURE_OPENAI_EMBED_MODEL=text-embedding-3-large
EMBED_DIM=1536
EMBED_TIMEOUT=3

# Langfuse (Optional)
LANGFUSE_PUBLIC_KEY=pk-lf-local
LANGFUSE_SECRET_KEY=sk-lf-local
LANGFUSE_HOST=http://localhost:3000

# Redis (Optional, falls back to in-memory)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# PostgreSQL for HITL (Optional)
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=chatmycv
POSTGRES_USER=chatmycv_user
POSTGRES_PASSWORD=chatmycv_pass
```

### Configuration Validation

Check your configuration:

```bash
python backend/config.py
```

Expected output:
```
CONFIGURATION SUMMARY
==============================================================
App:          ChatMyCV v0.2.0
Debug:        False

Azure OpenAI:
  Endpoint:   https://your-resource.openai.azure.com/
  LLM Model:  gpt-4.1-mini
  Embed Model: text-embedding-3-large

Langfuse:
  Enabled:    True
  Host:       http://localhost:3000

Redis:
  Enabled:    True
  Host:       localhost:6379

PostgreSQL:
  Enabled:    True
  Database:   chatmycv
==============================================================

✓ Configuration is valid
```

---

## Verification

### 1. Component Health Check

```bash
curl http://localhost:8000/observability/health
```

Expected response:
```json
{
  "status": "healthy",
  "components": {
    "langfuse": {
      "available": true,
      "enabled": true
    },
    "redis": {
      "available": true,
      "connected": true
    },
    "postgres": {
      "available": true,
      "connected": true
    }
  }
}
```

### 2. Test Chat with Tracing

Send a chat request:

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "lang": "en",
    "query": "Hello!",
    "character": "hr"
  }'
```

Response should include:
- `trace_id`: Langfuse trace identifier
- `route`: "chat" (detected as greeting)

### 3. Verify Langfuse Trace

1. Go to http://localhost:3000
2. Click "Traces" in sidebar
3. Find your trace by query text
4. Click to view spans: memory-load, router-decision, etc.

### 4. Test Memory Persistence

Send two messages in same session:

```bash
# First message
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "lang": "en",
    "query": "What is the candidate'\''s experience?",
    "session_id": "test-session-123",
    "character": "hr"
  }'

# Second message referencing first
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "lang": "en",
    "query": "What did I just ask?",
    "session_id": "test-session-123",
    "character": "hr"
  }'
```

Second response should:
- `route`: "memory"
- `response`: Summary of previous conversation

### 5. Test HITL Trigger

Send a low-confidence query:

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "lang": "en",
    "query": "Tell me about the legal policy claims.",
    "character": "hr"
  }'
```

Response should include:
- `hitl_triggered`: true
- `hitl_reason`: Explanation (e.g., "Risky keyword detected: legal")
- `hitl_review_id`: Review ID for tracking

Check pending reviews:

```bash
curl http://localhost:8000/hitl/pending
```

### 6. Run Prompt Evaluation

```bash
python backend/evaluation/run_eval.py --lang en --character hr
```

Expected output:
```
EVALUATION SUMMARY
==============================================================
Total Questions:    10
Average Score:      78.50%
Average Time:       2.34s
Result:             PASSED ✓
==============================================================
```

---

## Troubleshooting

### Langfuse Not Connected

**Symptoms**:
- `langfuse.enabled: false` in health check
- No `trace_id` in responses

**Solutions**:
1. Check Langfuse service is running: `docker ps | grep langfuse`
2. Verify credentials in `.env` match Langfuse UI (Settings → API Keys)
3. Check network connectivity: `curl http://localhost:3000/api/health`

### Redis Connection Failed

**Symptoms**:
- `redis.connected: false` in health check
- Warning: "Using in-memory conversation store"

**Solutions**:
1. Check Redis is running: `redis-cli ping`
2. Verify port is not blocked: `telnet localhost 6379`
3. Check `.env` has correct REDIS_HOST and REDIS_PORT

### PostgreSQL Connection Failed

**Symptoms**:
- `postgres.connected: false` in health check
- Warning: "HITL features will be disabled"

**Solutions**:
1. Check PostgreSQL is running: `pg_isready -h localhost -p 5433`
2. Verify database exists: `psql -U chatmycv_user -d chatmycv -c "SELECT 1;"`
3. Check credentials in `.env` match database user
4. Verify schema loaded: `psql -U chatmycv_user -d chatmycv -f backend/db/schema.sql`

### Docker Compose Issues

**Port conflicts**:
```bash
# Check what's using port 3000
netstat -ano | findstr :3000  # Windows
lsof -i :3000                  # Linux/macOS

# Stop conflicting service or change port in docker-compose.yml
```

**Services not starting**:
```bash
# View logs
docker-compose logs langfuse
docker-compose logs redis
docker-compose logs hitl-db

# Restart services
docker-compose restart
```

### Module Import Errors

**Symptoms**:
```
ImportError: No module named 'langfuse'
```

**Solutions**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify installation
pip list | grep langfuse
pip list | grep redis
pip list | grep psycopg2
```

---

## Production Considerations

### Security

1. **Change Default Passwords**:
   - Update Langfuse admin password
   - Use strong PostgreSQL credentials
   - Set Redis password

2. **Environment Variables**:
   - Never commit `.env` to git
   - Use secrets management (Azure Key Vault, AWS Secrets Manager)
   - Restrict access to configuration files

3. **Network Security**:
   - Use TLS for all connections
   - Restrict PostgreSQL to localhost or VPN
   - Enable Redis authentication

### Performance

1. **Redis Tuning**:
   ```
   # redis.conf
   maxmemory 2gb
   maxmemory-policy allkeys-lru
   ```

2. **PostgreSQL Indexing**:
   - Indexes already created in `schema.sql`
   - Monitor query performance
   - Consider partitioning for high volume

3. **Langfuse Optimization**:
   - Use async flushing in production
   - Batch trace uploads
   - Configure retention policies

### Monitoring

1. **Health Checks**:
   - Set up monitoring for `/observability/health`
   - Alert on component failures
   - Track response times

2. **Metrics**:
   - Monitor HITL trigger rate
   - Track average similarity scores
   - Measure LLM cost and latency

3. **Logs**:
   - Centralize logs (ELK, Splunk, etc.)
   - Set appropriate log levels
   - Rotate log files

### Scaling

1. **Redis**:
   - Use Redis Cluster for high availability
   - Replicate for read-heavy workloads
   - Consider Redis Sentinel for failover

2. **PostgreSQL**:
   - Enable replication for HA
   - Use connection pooling (PgBouncer)
   - Partition large tables

3. **Langfuse**:
   - Scale horizontally with load balancer
   - Use separate ClickHouse cluster
   - Optimize database queries

---

## Next Steps

Once your LLMOps infrastructure is running:

1. **Explore Langfuse**: View traces, analyze LLM calls, track costs
2. **Test HITL Workflow**: Trigger reviews, approve/reject via API
3. **Run Evaluations**: Establish baseline scores before prompt changes
4. **Integrate Streamlit**: Update `streamlit_app.py` to use new features
5. **Monitor Production**: Set up alerts for HITL triggers and errors

For more details, see:
- `CLAUDE.md` - Full architecture documentation
- `backend/evaluation/` - Prompt evaluation examples
- `docker-compose.yml` - Service configuration
- Langfuse docs: https://langfuse.com/docs

---

## Support

For issues or questions:
- Check `CLAUDE.md` for architecture details
- Review logs in `docker-compose logs`
- File issues at: https://github.com/anthropics/claude-code/issues
