# AI Business Analyst

An AI agent system with **two modes in one platform**:

- **Chatbot** — free-form Q&A on PDF, CSV, and TXT files (existing layer, always kept)
- **Business Analyst** — structured tasks that produce requirements, summaries, BRD drafts, etc. (added on top)

**Full documentation:** see [PROJECT.md](./PROJECT.md)

## Quick start

```bash
cd backend
pip install -r requirements.txt
pip install chromadb sentence-transformers langgraph langchain-text-splitters

# Create backend/.env with GOOGLE_API_KEY=...

python main.py
```

API runs at `http://localhost:8080`.

## Current capabilities

- Upload PDF, CSV, TXT files
- Q&A on documents via Agentic RAG (with retrieval quality loop)
- QMulti-turn session memory
- &A on CSV data via Pandas agent
- 



## Architecture

Dual-mode platform built as one complete agent stack:

- **Chat agents** — RAG + CSV (`create_agent`, tools, memory)
- **Task agents** — requirements, summarize, risks (`create_agent` + structured output)
- **Orchestrator** — LangGraph multi-step BA workflows

Full design, agent patterns, and build checklist: [PROJECT.md](./PROJECT.md)