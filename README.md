# CPE Course Builder Agent - can build a course on it own

5-node LangGraph pipeline that generates a complete course package from a topic, audience, and duration.

## Pipeline
```
Input → [Objectives] → [Outline] → [Script] → [Quiz] → [Gap Analysis] → Output
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure Ollama Cloud (copy .env.example → .env)
cp .env.example .env
# Edit .env and set OLLAMA_API_KEY from https://ollama.com/settings/keys

# 3. Run
python app.py
```

Visit: http://localhost:8000

## Stack
- **FastAPI** — backend + SSE streaming
- **LangGraph** — 5-node agentic pipeline
- **Ollama Cloud** — gemma4:31b-cloud inference
- **Vanilla HTML/CSS/JS** — no build step, runs instantly
