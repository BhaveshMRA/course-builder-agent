import os
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ollama import Client
from typing import TypedDict
from langgraph.graph import StateGraph, END

load_dotenv()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

client = Client(
    host=OLLAMA_HOST,
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {},
)


# ─── State ────────────────────────────────────────────────────────────────────

class CPEState(TypedDict):
    topic: str
    audience: str
    duration: str
    objectives: str
    outline: str
    script: str
    quiz: str
    gaps: str


# ─── Prompts ──────────────────────────────────────────────────────────────────

PROMPTS = {
    "objectives": """You are a senior instructional designer for professional graduate education.

Generate 6 clear, measurable learning objectives for:
Topic: {topic}
Audience: {audience}
Duration: {duration}

Rules:
- Each objective must start with a strong Bloom's Taxonomy action verb
- Be specific, measurable, and achievable within the given duration
- Tailor complexity to the audience level
- Number each objective

Format: numbered list only, no preamble.""",

    "outline": """You are a senior instructional designer.

Learning Objectives already defined:
{objectives}

Now build a detailed week-by-week module outline for:
Topic: {topic} | Audience: {audience} | Duration: {duration}

For EACH module include:
- Module title
- 3-4 key topics covered
- 1 hands-on activity or exercise
- Estimated time per section

Format cleanly with Module headers. No preamble.""",

    "script": """You are an experienced educator writing lecture content for professional learners.

Course: {topic}
Audience: {audience}
Outline:
{outline}

Write a complete lecture script for MODULE 1 only. Include:
- Engaging opening hook (real-world problem or story)
- Clear concept explanations with analogies
- 2-3 concrete industry examples
- Natural transitions between topics
- Closing summary and preview of next module

Tone: conversational but authoritative. No preamble.""",

    "quiz": """You are an assessment specialist designing evaluations for professional education.

Course: {topic}
Learning Objectives:
{objectives}

Create 5 multiple choice questions covering the first 2 modules. For each question:
Q[N]: [Question text]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: [Letter]
Rationale: [1-sentence explanation]

Questions must test application and understanding, not just recall. No preamble.""",

    "gaps": """You are a curriculum quality analyst reviewing a course design.

Course: {topic}
Audience: {audience}
Duration: {duration}

Objectives:
{objectives}

Outline:
{outline}

Identify exactly:
1. CONTENT GAPS — important topics or skills missing from this course
2. PREREQUISITE GAPS — knowledge assumed but not addressed
3. INDUSTRY ALIGNMENT — real-world skills or tools not reflected
4. RECOMMENDED ADDITIONS — specific additions with clear rationale

Be direct, specific, and actionable. No filler. No preamble."""
}


# ─── LLM Call ─────────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7, "num_predict": 2048},
    )
    return response.message.content.strip()


# ─── Nodes ────────────────────────────────────────────────────────────────────

def objectives_node(state: CPEState) -> CPEState:
    state["objectives"] = call_llm(PROMPTS["objectives"].format(**state))
    return state

def outline_node(state: CPEState) -> CPEState:
    state["outline"] = call_llm(PROMPTS["outline"].format(**state))
    return state

def script_node(state: CPEState) -> CPEState:
    state["script"] = call_llm(PROMPTS["script"].format(**state))
    return state

def quiz_node(state: CPEState) -> CPEState:
    state["quiz"] = call_llm(PROMPTS["quiz"].format(**state))
    return state

def gaps_node(state: CPEState) -> CPEState:
    state["gaps"] = call_llm(PROMPTS["gaps"].format(**state))
    return state


# ─── Graph ────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(CPEState)
    g.add_node("gen_objectives", objectives_node)
    g.add_node("gen_outline",    outline_node)
    g.add_node("gen_script",     script_node)
    g.add_node("gen_quiz",       quiz_node)
    g.add_node("gen_gaps",       gaps_node)

    g.set_entry_point("gen_objectives")
    g.add_edge("gen_objectives", "gen_outline")
    g.add_edge("gen_outline",    "gen_script")
    g.add_edge("gen_script",     "gen_quiz")
    g.add_edge("gen_quiz",       "gen_gaps")
    g.add_edge("gen_gaps",       END)

    return g.compile()

pipeline = build_graph()


# ─── API ──────────────────────────────────────────────────────────────────────

class CourseInput(BaseModel):
    topic: str
    audience: str
    duration: str


@app.get("/")
async def serve():
    return FileResponse("static/index.html")


@app.post("/generate")
async def generate(data: CourseInput):
    async def stream():
        state: CPEState = {
            "topic":      data.topic,
            "audience":   data.audience,
            "duration":   data.duration,
            "objectives": "",
            "outline":    "",
            "script":     "",
            "quiz":       "",
            "gaps":       "",
        }

        steps = [
            ("objectives", "Learning Objectives",  objectives_node),
            ("outline",    "Module Outline",        outline_node),
            ("script",     "Lecture Script",        script_node),
            ("quiz",       "Quiz Questions",        quiz_node),
            ("gaps",       "Content Gap Analysis",  gaps_node),
        ]

        for key, label, node_fn in steps:
            yield f"data: {json.dumps({'type': 'progress', 'step': key, 'label': label})}\n\n"
            await asyncio.sleep(0.05)

            try:
                state = node_fn(state)
                yield f"data: {json.dumps({'type': 'content', 'step': key, 'content': state[key]})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'step': key, 'message': str(e)})}\n\n"
                return

            await asyncio.sleep(0.05)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
