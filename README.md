# research-agent-crew

A LangGraph multi-agent research system that takes a query, autonomously plans, searches the web, summarises, fact-checks, and drafts a polished report — then pauses for **human-in-the-loop review** before finalising.

## How It Works

```
POST /research (query)
        │
        ▼
  Planner Agent
  — breaks query into 3-5 focused sub-questions —
        │
        ▼
  Researcher Agent
  — Tavily web search per sub-question —
        │
        ▼
  Summariser Agent
  — synthesises all findings into coherent prose —
        │
        ▼
  Fact-Checker Agent
  — flags low-confidence claims —
        │
        ▼
  Editor Agent
  — writes 500-700 word polished report —
        │
        ▼
  ⏸ HUMAN REVIEW (awaiting_approval)
  GET /draft/{task_id}       → read draft
  POST /approve/{task_id}    → accept
  POST /reject/{task_id}     → reject with feedback → loops back to Editor
        │ (approved)
        ▼
  Finaliser Agent
  → report stored in Redis
        │
        ▼
  GET /report/{task_id}
```

## Quick Start

### Step 1 — Configure

```bash
cp .env.example .env
```

Edit `.env` — minimum required:
```env
GEMINI_API_KEY=AIza...      # Required
TAVILY_API_KEY=tvly-...     # Optional — mock results used if blank
```

### Step 2 — Run (Docker)

```bash
docker compose up --build
```

API available at `http://localhost:8002`

### Step 3 — Run (local, no Docker)

```bash
pip install -r requirements.txt

# Start Redis separately (or use Docker just for Redis)
docker run -d -p 6379:6379 redis:7-alpine

uvicorn api.main:app --reload --port 8002
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/research` | Start a research task. Body: `{"query": "..."}` |
| `GET` | `/status/{task_id}` | Poll task status and progress |
| `GET` | `/draft/{task_id}` | Get current draft (when `awaiting_approval`) |
| `POST` | `/approve/{task_id}` | Approve draft → generates final report |
| `POST` | `/reject/{task_id}` | Reject with feedback. Body: `{"feedback": "..."}` |
| `GET` | `/report/{task_id}` | Get final approved report (when `complete`) |
| `GET` | `/health` | Health check |

## Demo Walkthrough

```bash
# 1. Start a research task
curl -X POST http://localhost:8002/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest advances in LLM agents in 2024?"}'
# → {"task_id": "abc-123", "status": "running", ...}

# 2. Poll until awaiting_approval (takes 30-60s)
curl http://localhost:8002/status/abc-123

# 3. Read the draft
curl http://localhost:8002/draft/abc-123

# 4a. Approve
curl -X POST http://localhost:8002/approve/abc-123

# 4b. OR reject with feedback
curl -X POST http://localhost:8002/reject/abc-123 \
  -H "Content-Type: application/json" \
  -d '{"feedback": "Add more detail about OpenAI and Anthropic specifically"}'

# 5. Get final report
curl http://localhost:8002/report/abc-123
```

## Task Status Flow

```
queued → running → awaiting_approval ←──────────────┐
                         │                           │
                    (reject)                    (revising)
                         │                           │
                    (approve)          editor revises + pauses again
                         │
                    finalising
                         │
                      complete
```

## Project Structure

```
research-agent-crew/
├── agent/
│   ├── state.py         # ResearchState TypedDict
│   ├── tools.py         # Tavily search + mock fallback
│   ├── nodes.py         # All agent node functions
│   └── graph.py         # LangGraph StateGraph + compile
├── api/
│   └── main.py          # FastAPI app
├── prompts/             # System prompts for each agent
├── models.py            # Pydantic request/response models
├── store.py             # Redis task store
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Features

- **5 specialist agents**: Planner → Researcher → Summariser → Fact-Checker → Editor
- **Human-in-the-loop**: Draft pauses for review before finalising (`interrupt_before` in LangGraph)
- **Iterative revision**: Reject with feedback → Editor revises → review again (max 3 cycles)
- **Tavily search**: Real web search with graceful mock fallback if no API key
- **Redis persistence**: Task state survives restarts; 7-day TTL
- **Gemini 2.5 Flash**: All agents use the same model via OpenAI-compatible endpoint

## Author

**Sanidhya Singh** — AI/ML Engineer  
[GitHub](https://github.com/sanidhya-ai-ml) · [LinkedIn](https://www.linkedin.com/in/sanidhya-aiml)
