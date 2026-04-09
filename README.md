# kb-mocker

A FastAPI-based AI agent that answers questions by autonomously searching a local markdown knowledge base using LangChain tool calling and Claude via OpenRouter.

## How It Works

1. User sends a question to `POST /api/v1/ask`
2. An agentic loop starts — the LLM decides which knowledge files to read
3. The agent calls `list_knowledge_files()` and `load_knowledge(filename)` tools as needed
4. Once the agent has enough context, it stops and returns a structured answer

```
POST /api/v1/ask
    │
    ▼
run_agent(question)
    ├── LLM + Tools initialized
    └── Agentic loop (up to max_iterations):
        ├── LLM reasons over messages
        ├── tool_calls → list_knowledge_files() or load_knowledge(filename)
        └── stop → format_response() → QuestionResponse
```

## Project Structure

```
kb-mocker/
├── main.py                     # FastAPI app entry point
├── pyproject.toml              # Project metadata and dependencies
├── .env.example                # Example environment config
├── .knowledge/                 # Markdown knowledge base files
└── src/kb_mocker/
    ├── config.py               # Pydantic settings (loaded from .env)
    ├── api/routes.py           # POST /api/v1/ask endpoint
    ├── chains/qa_chain.py      # Agentic loop and response formatting
    └── tools/knowledge.py      # list_knowledge_files and load_knowledge tools
```

## Setup

**Requirements:** Python 3.13, [uv](https://docs.astral.sh/uv/)

```bash
# 1. Clone and install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env — set your OPENROUTER_API_KEY at minimum
```

### Environment Variables

| Variable               | Default                              | Description                          |
|------------------------|--------------------------------------|--------------------------------------|
| `OPENROUTER_API_KEY`   | —                                    | **Required.** Your OpenRouter API key |
| `MODEL_NAME`           | `anthropic/claude-sonnet-4-6`        | Model to use via OpenRouter          |
| `OPENROUTER_BASE_URL`  | `https://openrouter.ai/api/v1`       | OpenRouter API base URL              |
| `KNOWLEDGE_BASE_PATH`  | `.knowledge`                         | Path to markdown knowledge files     |
| `MAX_ITERATIONS`       | `10`                                 | Max agentic loop iterations          |

## Running the Project

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

For development with auto-reload:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server starts at `http://0.0.0.0:8000`.

## Usage

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Claude Managed Agents?"}'
```

Response:

```json
{
  "answer": "Claude Managed Agents is a suite of composable APIs..."
}
```

## Adding Knowledge

Drop any `.md` file into `.knowledge/`. The agent will automatically discover and use it — no reindexing required.

## Tech Stack

- **FastAPI** — HTTP API
- **LangChain + LangGraph** — Agentic tool-calling loop
- **OpenRouter** — LLM API gateway (default: Claude Sonnet 4.6)
- **Pydantic** — Settings and structured output
- **uv** — Dependency management
