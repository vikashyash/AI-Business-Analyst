# AI Business Analyst — Project Documentation

> An AI-powered system that automates junior Business Analyst (BA) work using intelligent agents.  
> Built as a portfolio project and a complete hands-on learning path for agentic AI.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Vision & Goals](#2-vision--goals)
3. [What a Business Analyst Does](#3-what-a-business-analyst-does)
4. [Dual-Mode Design: Chatbot + BA](#4-dual-mode-design-chatbot--ba)
5. [System Architecture](#5-system-architecture)
6. [Project Structure](#6-project-structure)
7. [Tech Stack](#7-tech-stack)
8. [API Reference](#8-api-reference)
9. [Core Components](#9-core-components)
10. [Agent Design](#10-agent-design)
11. [Data Flow](#11-data-flow)
12. [Configuration](#12-configuration)
13. [Setup & Running](#13-setup--running)
14. [Complete Build Checklist](#14-complete-build-checklist)
15. [Known Issues & Limitations](#15-known-issues--limitations)
16. [Agent Concepts You Learn](#16-agent-concepts-you-learn)

---



## 1. Overview

**AI Business Analyst** is a backend application that ingests business documents (PDF, CSV, TXT), indexes them for retrieval, and runs multiple types of AI agents to support BA work.

The system is a **dual-mode platform**:

1. **Chatbot mode** — Free-form Q&A on documents (`POST /analyze`).
2. **BA mode** — Structured tasks that produce saved deliverables (`POST /workspaces/{id}/tasks` and `/run`).

Both modes share the same agent engine. BA is **added on top** of chat — nothing is replaced.

### Design philosophy 

> **Build the full agent stack once. Learn every core agent concept in this single project.**

There is no phased upgrade plan. Every layer — worker agents, task agents, tools, structured outputs, memory, and orchestrator — is part of one unified architecture from the start.

### Implementation status


| Capability                                      | Status         |
| ----------------------------------------------- | -------------- |
| Upload PDF, CSV, TXT files                      | ✅ Implemented  |
| List uploaded documents                         | ✅ Implemented  |
| Q&A on PDF/Text via Agentic RAG agent           | ✅ Implemented  |
| Q&A on CSV via Pandas agent                     | ✅ Implemented  |
| Multi-turn chat memory                          | ✅ Implemented  |
| Agentic retrieval (grade + rewrite loop)        | ✅ Implemented  |
| Workspaces (project container)                  | 🔲 In progress |
| Shared agent tools (`tools.py`)                 | 🔲 To build    |
| BA task agents (requirements, summarize, risks) | 🔲 To build    |
| LangGraph orchestrator (`/run`)                 | 🔲 To build    |
| Deliverable storage per workspace               | 🔲 To build    |


---



## 2. Vision & Goals



### Primary goals

1. **Portfolio project** — Demonstrate a complete agentic system: RAG, tools, structured outputs, multi-agent orchestration, memory.
2. **Practical learning** — Learn all major agent concepts by building one real BA product, not toy examples spread across phases.
3. **Business value** — Automate junior BA work: document analysis, requirements extraction, data profiling, report generation.



### End-state (built as one system)

```
┌─────────────────────────────────────────────────────────────┐
│                      Workspace (Project)                     │
│  Documents: budget.pdf, sales.csv, meeting-notes.txt        │
├──────────────────────────┬──────────────────────────────────┤
│     CHATBOT MODE         │           BA MODE                 │
│  POST /analyze           │  POST /workspaces/{id}/tasks      │
│  Free-form questions     │  Task agents → deliverables       │
│  Multi-turn memory       │  Orchestrator for multi-step jobs │
└──────────────────────────┴──────────────────────────────────┘
                              │
                    Shared agent engine
```

**Chatbot mode:** explore documents freely with conversational agents.

**BA mode:** run task agents that use tools, produce structured JSON deliverables, and save them to the workspace.

**Orchestrator:** LangGraph supervisor that chains multiple task agents for goals like "analyze all files and draft a leadership brief."

---



## 3. What a Business Analyst Does

A junior Business Analyst bridges business stakeholders and technical teams.


| BA activity                | What the agent system does               |
| -------------------------- | ---------------------------------------- |
| Read reports and data      | RAG agent + CSV agent search and analyze |
| Extract requirements       | Requirements task agent                  |
| Summarize for leadership   | Summarize task agent                     |
| Find risks and assumptions | Risks task agent                         |
| Profile datasets           | Profile task agent (CSV tool)            |
| Draft BRD / brief          | Deliverable agent + orchestrator         |
| Track project artifacts    | Workspace deliverable storage            |


**Target:** Automate 70–80% of junior BA document and data work.

---



## 4. Dual-Mode Design: Chatbot + BA



### Design principle

> **Keep the chatbot. Add BA task agents. Share tools underneath.**


| Aspect   | Chatbot mode                | BA mode                               |
| -------- | --------------------------- | ------------------------------------- |
| Entry    | `POST /analyze`             | `POST /workspaces/{id}/tasks`         |
| Input    | Free-text `question`        | `task_type` + `document_ids`          |
| Who runs | Chat agent (RAG or CSV)     | Task agent (LangChain `create_agent`) |
| Output   | Plain `answer` string       | Structured JSON deliverable (saved)   |
| Memory   | Session thread per doc      | Session thread per task               |
| Use case | Explore, clarify, follow-up | Formal BA outputs                     |


Both modes call the **same shared tools** (`search_document`, `analyze_csv`) defined in `agents/tools.py`.

### Example user journey

1. Create workspace, upload budget PDF + sales CSV
2. **Chat:** "What are the top 3 cost drivers?" → RAG agent answers
3. **BA task:** `extract_requirements` → requirements task agent searches docs, returns structured JSON, saves deliverable
4. **Chat:** "Explain requirement REQ-3" → same session context
5. **BA run:** "Summarize everything and draft a leadership brief" → orchestrator chains task agents

---



## 5. System Architecture



### Full architecture (single build target)

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       ▼
┌──────────────────────────────────────────────────────────────┐
│                         FastAPI (main.py)                     │
│  /upload  /documents  /analyze          ← CHAT (kept)       │
│  /workspaces  /tasks  /run              ← BA + orchestrator  │
└──────┬───────────────────────────────────────────────────────┘
       │
       ├─────────────────────────┬────────────────────────────┐
       ▼                         ▼                            ▼
┌─────────────┐         ┌─────────────────┐         ┌──────────────────┐
│ Chat Router │         │  Task Router    │         │  Orchestrator    │
│ analyze.py  │         │  tasks/run_task │         │  orchestrator.py │
└──────┬──────┘         └────────┬────────┘         │  (LangGraph)     │
       │                           │                 └────────┬─────────┘
       ▼                           ▼                          │
┌─────────────┐         ┌─────────────────┐                   │
│ Chat Agents │         │  Task Agents    │◄──────────────────┘
│ RAG + CSV   │         │  requirements   │
└──────┬──────┘         │  summarize      │
       │                │  find_risks     │
       │                │  profile_data   │
       │                │  generate_brd   │
       └────────┬───────┴────────┬────────┘
                ▼                ▼
       ┌─────────────────────────────────┐
       │      Shared Tools (tools.py)     │
       │  search_document  |  analyze_csv │
       └────────────────┬────────────────┘
                        ▼
       ┌─────────────────────────────────┐
       │  Agentic Retrieval Loop          │
       │  grade → rewrite → retry         │
       └────────────────┬────────────────┘
                        ▼
       ┌─────────────────────────────────┐
       │  ChromaDB + manifest + workspaces│
       └─────────────────────────────────┘
```



### Agent hierarchy (4 layers — all built together)

```
Layer 4 — Orchestrator Agent     LangGraph StateGraph — plans and chains tasks
Layer 3 — Task Agents            create_agent + tools — one per BA deliverable
Layer 2 — Chat Agents            create_agent + tools — free Q&A (RAG, CSV)
Layer 1 — Shared Tools           @tool functions — search_document, analyze_csv
Layer 0 — Retrieval sub-loop     grade + rewrite + retry (inside search tool)
```

**Only Layer 0's indexing step (upload → chunk → embed) is a fixed pipeline.** Everything above it is agentic.

---



## 6. Project Structure



### Current + complete target

```
AI-Business-Analyst/
├── PROJECT.md
├── README.md
├── Uploads/
│   ├── manifest.json
│   └── {uuid}.pdf|.csv|.txt
├── workspaces/                          # runtime workspace data
│   └── {ws_id}/deliverables/
│
└── backend/
    ├── main.py                          # all API routes
    ├── requirements.txt
    └── app/
        ├── config.py
        └── api/
            ├── analyze.py               # chat router
            ├── services.py              # file parsing utilities
            │
            ├── agents/
            │   ├── agentic_rag_retrieval.py   # chat RAG agent ✅
            │   ├── csv_agent.py               # chat CSV agent ✅
            │   ├── tools.py                   # shared @tool functions
            │   ├── requirements_agent.py      # BA task agent
            │   ├── summarize_agent.py         # BA task agent
            │   ├── risks_agent.py             # BA task agent
            │   ├── profile_agent.py           # BA task agent
            │   ├── deliverable_agent.py       # BRD / brief generator
            │   └── orchestrator.py            # LangGraph supervisor
            │
            ├── tasks/
            │   └── run_task.py                # task_type → agent dispatch
            │
            ├── workspaces/
            │   ├── models.py                    # Pydantic request bodies ✅
            │   └── workspace_store.py           # CRUD + deliverables ✅
            │
            └── kb_handler/
                ├── kb_knowledge_file_store.py
                └── kb_insertion.py
```

---



## 7. Tech Stack


| Layer             | Technology                                | Used for                        |
| ----------------- | ----------------------------------------- | ------------------------------- |
| API               | FastAPI + Uvicorn                         | REST endpoints                  |
| LLM               | Google Gemini (`langchain-google-genai`)  | All agents and graders          |
| Chat agents       | LangChain `create_agent`                  | RAG chat, CSV chat              |
| Task agents       | LangChain `create_agent` + `@tool`        | BA deliverable tasks            |
| Orchestrator      | LangGraph `StateGraph`                    | Multi-step BA workflows         |
| Memory            | LangGraph `InMemorySaver`                 | Chat + task session threads     |
| Structured output | Pydantic + `with_structured_output`       | Requirements, summaries, grades |
| Embeddings        | `sentence-transformers/all-mpnet-base-v2` | Vector search                   |
| Vector DB         | ChromaDB                                  | Per-document chunk storage      |
| CSV               | LangChain Pandas agent                    | Code-executing data analysis    |
| PDF               | pdfplumber                                | Text extraction                 |




### Required dependencies

```
fastapi uvicorn python-multipart python-dotenv
langchain langchain-core langchain-experimental langchain-google-genai
langchain-text-splitters langgraph
chromadb sentence-transformers
pdfplumber pandas pydantic
```

---



## 8. API Reference

Base URL: `http://localhost:8080`

### Chatbot endpoints (implemented)


| Method | Path         | Purpose                       |
| ------ | ------------ | ----------------------------- |
| POST   | `/upload`    | Upload file                   |
| GET    | `/documents` | List all documents            |
| POST   | `/analyze`   | Free-form Q&A on one document |




### Workspace + BA endpoints (full build)


| Method | Path                       | Mode     | Purpose                        |
| ------ | -------------------------- | -------- | ------------------------------ |
| POST   | `/workspaces`              | Setup    | Create project                 |
| GET    | `/workspaces`              | Setup    | List projects                  |
| GET    | `/workspaces/{id}`         | Both     | Project details + deliverables |
| POST   | `/workspaces/{id}/upload`  | Both     | Upload file into project       |
| POST   | `/workspaces/{id}/analyze` | **Chat** | Q&A scoped to workspace        |
| POST   | `/workspaces/{id}/tasks`   | **BA**   | Run one task agent             |
| POST   | `/workspaces/{id}/run`     | **BA**   | Orchestrator multi-step goal   |




### When to use which


| User intent                                 | Endpoint                        |
| ------------------------------------------- | ------------------------------- |
| Quick question on one file                  | `POST /analyze`                 |
| Chat inside a project                       | `POST /workspaces/{id}/analyze` |
| Extract requirements, summarize, find risks | `POST /workspaces/{id}/tasks`   |
| "Analyze everything and write a brief"      | `POST /workspaces/{id}/run`     |




### `POST /workspaces/{id}/tasks` — BA task request

```json
{
  "task_type": "extract_requirements",
  "document_ids": ["doc-uuid-1"],
  "session_id": "ba-session-1",
  "params": { "focus": "functional requirements" }
}
```



### `POST /workspaces/{id}/tasks` — BA task response

```json
{
  "workspace_id": "ws-uuid",
  "task_type": "extract_requirements",
  "session_id": "ba-session-1",
  "result": {
    "requirements": [
      { "id": "REQ-1", "text": "...", "type": "functional", "source_hint": "page 3" }
    ],
    "open_questions": ["Who approves the budget?"]
  },
  "saved_as": "workspaces/ws-uuid/deliverables/extract_requirements_20260621.json"
}
```



### Supported task types


| task_type              | Agent file              | Tools used                       | Output                   |
| ---------------------- | ----------------------- | -------------------------------- | ------------------------ |
| `extract_requirements` | `requirements_agent.py` | `search_document`                | Requirements JSON        |
| `summarize`            | `summarize_agent.py`    | `search_document`, `analyze_csv` | Executive summary JSON   |
| `find_risks`           | `risks_agent.py`        | `search_document`                | Risks + assumptions JSON |
| `profile_data`         | `profile_agent.py`      | `analyze_csv`                    | Data profile JSON        |
| `generate_brd`         | `deliverable_agent.py`  | all tools                        | BRD draft JSON           |


---



## 9. Core Components



### 9.1 Document Store (`kb_knowledge_file_store.py`)

- Validates and saves uploads
- Maintains `manifest.json`
- Triggers Chroma indexing for PDF/Text
- Optional `workspace_id` on each document



### 9.2 Knowledge Base Insertion (`kb_insertion.py`)

Fixed indexing pipeline (not agentic — by design):

```
parse → clean → chunk → embed → store in Chroma
```



### 9.3 Workspace Store (`workspace_store.py`)

- CRUD for projects
- Attach documents to workspace
- Save deliverable JSON files



### 9.4 Task Router (`tasks/run_task.py`)

Simple dispatch — no over-engineering:

```python
HANDLERS = {
    "extract_requirements": requirements_agent.run,
    "summarize": summarize_agent.run,
    "find_risks": risks_agent.run,
    "profile_data": profile_agent.run,
    "generate_brd": deliverable_agent.run,
}
```

Each handler invokes a **task agent**, not a plain LLM chain.

### 9.5 Chat Router (`analyze.py`)

Routes by file type to chat agents:

```
CSV       → csv_agent
PDF/Text  → agentic_rag_retrieval
```

---



## 10. Agent Design

This is the most important section. Every BA task is a **real agent**, not a plain LLM call.

### 10.1 What is an agent vs a pipeline in this project?


| Component                             | Type                 | Why                                                     |
| ------------------------------------- | -------------------- | ------------------------------------------------------- |
| KB indexing (`kb_insertion.py`)       | **Pipeline**         | Fixed steps, no LLM decisions                           |
| Chat RAG (`agentic_rag_retrieval.py`) | **Agent**            | LLM decides when/how to call `search_document`          |
| Chat CSV (`csv_agent.py`)             | **Agent**            | LLM generates and runs Python on DataFrame              |
| BA task agents (`*_agent.py`)         | **Agent**            | LLM decides how many searches, which queries, when done |
| Orchestrator (`orchestrator.py`)      | **Agent**            | LangGraph plans and chains task agents                  |
| Retrieval quality loop                | **Agentic sub-loop** | Grade → rewrite → retry inside the search tool          |




### 10.2 Shared Tools (`agents/tools.py`)

All agents share the same tools:

```python
@tool
def search_document(query: str) -> str:
    """Search PDF/Text doc with quality-checked retrieval."""
    # calls retrieve_with_quality_loop internally

@tool
def analyze_csv(question: str) -> str:
    """Run pandas agent on the workspace CSV file."""
    # calls csv_agent internally
```

Tools are factories when bound to a specific doc:

```python
def make_tools(llm, doc_id, embed_model, csv_path=None):
    # returns [search_document] or [search_document, analyze_csv]
```



### 10.3 Chat RAG Agent (implemented)

```
User question
    → LangChain create_agent
    → calls search_document tool
    → retrieve_with_quality_loop (grade + rewrite + retry)
    → LLM composes answer
    → LangGraph InMemorySaver (session memory)
```

Framework: `create_agent` + `@tool` + `InMemorySaver`

### 10.4 Chat CSV Agent (implemented)

```
User question
    → create_pandas_dataframe_agent
    → LLM writes Python → executes on DataFrame
    → InMemoryChatMessageHistory (session memory)
```

Framework: LangChain experimental pandas agent

### 10.5 BA Task Agents (to build — all follow same pattern)

Every task agent uses `create_agent` **+ tools + structured final output**.

Example: Requirements Agent

```
Task invoked (extract_requirements)
    → create_agent with search_document tool(s)
    → Agent autonomously:
        1. Searches for "functional requirements"
        2. Searches again for "constraints and acceptance criteria" if needed
        3. Compiles findings
    → Final step: with_structured_output(RequirementsOutput)
    → Save JSON to workspace deliverables/
```

Example: Summarize Agent

```
Task invoked (summarize)
    → create_agent with search_document + analyze_csv tools
    → Agent searches docs AND queries CSV as needed
    → Structured summary output
    → Save deliverable
```

**Why agents for tasks (not plain LLM chains)?**

- LLM decides **how many searches** to run
- LLM picks **different query angles** (risks, KPIs, stakeholders)
- LLM can **combine doc + CSV** sources dynamically
- You learn tool-calling, agent loops, and structured output in one place



### 10.6 Task Agent template (use for every BA task)

```python
# agents/requirements_agent.py

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from .tools import make_tools

_checkpointer = InMemorySaver()

class RequirementsOutput(BaseModel):
    requirements: list[dict]
    open_questions: list[str]

STRUCTURE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Convert the agent research below into structured requirements."),
    ("human", "{research}"),
])

def _run_sync(llm, embed_model, docs, params, session_id):
    tools = make_tools(llm, docs, embed_model)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a BA requirements extractor. "
            "Use search_document to find requirement passages. "
            "Search multiple times with different queries if needed. "
            "When done, summarize all findings."
        ),
        checkpointer=_checkpointer,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Extract all requirements from the documents."}]},
        config={"configurable": {"thread_id": f"req:{session_id}"}},
    )
    research = result["messages"][-1].content
    structured = (STRUCTURE_PROMPT | llm.with_structured_output(RequirementsOutput)).invoke({"research": research})
    return structured.model_dump()

async def run(llm, embed_model, docs, params, session_id="ba-task"):
    import asyncio
    return await asyncio.to_thread(_run_sync, llm, embed_model, docs, params, session_id)
```

Copy this pattern for `summarize_agent.py`, `risks_agent.py`, etc. — change the system prompt and output model.

### 10.7 Orchestrator Agent (`agents/orchestrator.py`)

Uses **LangGraph StateGraph** to chain task agents for open-ended goals.

```
User goal: "Analyze budget PDF and sales CSV, write leadership brief"
    │
    ▼
┌─────────────┐
│  plan_node  │  LLM breaks goal into task list
└──────┬──────┘
       ▼
┌─────────────┐
│ execute_node│  runs each task agent in sequence
└──────┬──────┘
       ▼
┌─────────────┐
│ synth_node  │  deliverable_agent combines all outputs
└──────┬──────┘
       ▼
  Final deliverable saved
```

Framework: LangGraph `StateGraph` + shared task agent handlers

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class OrchestratorState(TypedDict):
    ws_id: str
    goal: str
    plan: list[str]
    task_results: list[dict]
    final_deliverable: dict
```



### 10.8 Agent concepts map


| Concept                   | Where in this project              | Framework                           |
| ------------------------- | ---------------------------------- | ----------------------------------- |
| Tool calling              | All agents                         | LangChain `@tool`                   |
| ReAct agent loop          | Chat + task agents                 | `create_agent`                      |
| Structured output         | Task final step, retrieval grading | Pydantic + `with_structured_output` |
| Agent memory              | Chat + task sessions               | LangGraph `InMemorySaver`           |
| Agentic retrieval         | Inside `search_document` tool      | Custom grade/rewrite loop           |
| Code-executing agent      | CSV chat + `analyze_csv` tool      | Pandas agent                        |
| Multi-agent orchestration | `/run` endpoint                    | LangGraph `StateGraph`              |
| Task routing              | `tasks/run_task.py`                | Simple Python dict                  |
| RAG                       | Chroma + embeddings                | ChromaDB + sentence-transformers    |


---



## 11. Data Flow



### Chat flow

```
POST /analyze
  → analyze.py
  → RAG agent or CSV agent (create_agent)
  → tool calls → retrieval loop or pandas code
  → answer string
```



### BA task flow

```
POST /workspaces/{id}/tasks
  → run_task.py (dispatch by task_type)
  → task agent (create_agent + tools)
  → agent searches docs / analyzes CSV autonomously
  → structured output (Pydantic)
  → save_deliverable()
  → JSON response
```



### Orchestrator flow

```
POST /workspaces/{id}/run  { "goal": "..." }
  → orchestrator.py (LangGraph)
</StateGraph>)
  → plan → execute task agents → synthesize
  → save final deliverable
  → JSON response
```

---



## 12. Configuration

Create `backend/.env`:

```env
GOOGLE_API_KEY=your_key_here
LLM_MODEL=gemini-2.5-flash-lite
UPLOAD_DIR=../Uploads
KB=../Uploads/kb
WORKSPACE_PATH=../workspaces
MAX_UPLOAD_MB=10
```


| Variable         | Default                 | Purpose                |
| ---------------- | ----------------------- | ---------------------- |
| `GOOGLE_API_KEY` | —                       | Gemini API key         |
| `LLM_MODEL`      | `gemini-2.5-flash-lite` | Model for all agents   |
| `UPLOAD_DIR`     | `uploads`               | Uploaded files         |
| `KB`             | `kb`                    | Chroma stores          |
| `WORKSPACE_PATH` | `workspaces`            | Project + deliverables |


---



## 13. Setup & Running

```bash
cd backend
pip install -r requirements.txt
pip install chromadb sentence-transformers langgraph langchain-text-splitters
python main.py
```



### Full test flow

```bash
# 1. Create workspace
curl -X POST http://localhost:8080/workspaces \
  -H "Content-Type: application/json" \
  -d '{"name": "Budget Project"}'

# 2. Upload to workspace
curl -X POST http://localhost:8080/workspaces/{ws_id}/upload -F "file=@report.pdf"

# 3. Chat
curl -X POST http://localhost:8080/workspaces/{ws_id}/analyze \
  -H "Content-Type: application/json" \
  -d '{"document_id":"DOC_ID","session_id":"s1","question":"What are the key risks?"}'

# 4. BA task (task agent)
curl -X POST http://localhost:8080/workspaces/{ws_id}/tasks \
  -H "Content-Type: application/json" \
  -d '{"task_type":"extract_requirements","document_ids":["DOC_ID"],"session_id":"s1","params":{}}'

# 5. Orchestrator
curl -X POST http://localhost:8080/workspaces/{ws_id}/run \
  -H "Content-Type: application/json" \
  -d '{"goal":"Summarize all documents and draft a leadership brief","session_id":"s1"}'
```

---



## 14. Complete Build Checklist

One checklist — no phases. Build top to bottom.

### Foundation

- [x] FastAPI app with `/upload`, `/documents`, `/analyze`
- [x] Chat RAG agent with agentic retrieval loop
- [x] Chat CSV agent with session memory
- [x] Chroma indexing for PDF/Text
- [x] Workspace store (CRUD + deliverables)
- [ ] Fix `main.py` workspace routes (GET `/workspaces`, fix `ws_analyze`)
- [ ] Pin all dependencies in `requirements.txt`



### Shared agent layer

- [ ] Create `agents/tools.py` with `search_document` and `analyze_csv` tools
- [ ] Refactor RAG chat agent to use shared tool from `tools.py`



### BA task agents (each = `create_agent` + tools + structured output)

- [ ] `agents/requirements_agent.py`
- [ ] `agents/summarize_agent.py`
- [ ] `agents/risks_agent.py`
- [ ] `agents/profile_agent.py`
- [ ] `agents/deliverable_agent.py`
- [ ] `tasks/run_task.py` — dispatch dict + save deliverable
- [ ] `POST /workspaces/{id}/tasks` in `main.py`



### Orchestrator

- [ ] `agents/orchestrator.py` — LangGraph StateGraph
- [ ] `POST /workspaces/{id}/run` in `main.py`



### Optional polish

- [ ] Return citations in chat responses
- [ ] CORS for frontend
- [ ] Basic tests

---



## 15. Known Issues & Limitations


| Issue                           | Notes                                                                                                 |
| ------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Session memory in-memory only   | Lost on restart — acceptable for portfolio                                                            |
| No authentication               | Open API                                                                                              |
| No CORS                         | Add when building frontend                                                                            |
| CSV `allow_dangerous_code=True` | Dev only                                                                                              |
| `main.py` has route bugs        | `@app.post("list_workspaces")` should be `@app.get("/workspaces")`; `ws_analyze` calls wrong function |


---



## 16. Agent Concepts You Learn

This single project covers the full agent learning path:


| #   | Concept                                  | You build it in                 |
| --- | ---------------------------------------- | ------------------------------- |
| 1   | RAG (retrieval augmented generation)     | Chroma + embeddings             |
| 2   | Agentic retrieval (grade + rewrite loop) | `retrieve_with_quality_loop`    |
| 3   | Tool calling                             | `@tool` in `tools.py`           |
| 4   | ReAct agent (reason + act loop)          | Chat RAG agent, task agents     |
| 5   | Structured LLM output                    | Pydantic models on task results |
| 6   | Session / thread memory                  | LangGraph `InMemorySaver`       |
| 7   | Code-executing agent                     | Pandas CSV agent                |
| 8   | Task routing                             | `tasks/run_task.py`             |
| 9   | Specialized task agents                  | `requirements_agent.py`, etc.   |
| 10  | Multi-agent orchestration                | LangGraph orchestrator          |
| 11  | Dual-mode product design                 | Chat + BA in one system         |




### Portfolio talking points

1. **Full agent stack** — Not just chatbot; task agents and orchestrator included.
2. **Agentic RAG** — Retrieval quality loop with LLM grading and BA-aware query rewriting.
3. **Shared tool layer** — Chat agents and task agents reuse the same tools.
4. **Real BA outputs** — Structured deliverables saved per workspace.
5. **LangGraph orchestrator** — Multi-step goal planning and execution.

---



## Appendix: Quick Reference



### Agent vs pipeline in this project

```
Pipeline (fixed steps):     kb_insertion.py only
Agent (LLM decides):      everything else
```



### File responsibility

```
main.py              → routes
analyze.py           → chat dispatch
tasks/run_task.py    → BA task dispatch
agents/tools.py      → shared tools
agents/*_agent.py    → one agent per responsibility
agents/orchestrator.py → LangGraph multi-step
workspaces/          → project storage
kb_handler/          → files + vectors
```



### Mental model

```
CHAT:   question  →  chat agent  →  tools  →  answer
BA:     task_type →  task agent  →  tools  →  structured deliverable (saved)
RUN:    goal      →  orchestrator → task agents → final deliverable (saved)
```

---

*Last updated: June 2026*