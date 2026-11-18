"""FastAPI interface for Viet's Supervisor Agent (run with uvicorn src.api:app)."""
import os
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .vinh_agents import VinhAgent

app = FastAPI(title="Vinh – VietJobs NLP & RAG API", version="1.0.0")

@lru_cache(maxsize=1)
def agent():
    return VinhAgent()

class SkillsRequest(BaseModel):
    text: str = Field(min_length=1)
    top_k: int = Field(default=12, ge=1, le=50)
    threshold: float | None = Field(default=None, ge=0, le=1)

class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    use_llm: bool = False

@app.get("/health")
def health():
    return {"status": "ok", "model_configured": bool(os.environ.get("VINH_MODEL_DIR"))}

@app.post("/skills/extract")
def extract(body: SkillsRequest):
    try:
        return {"skills": agent().extract_skills(body.text, body.top_k, body.threshold)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/jobs/search")
def search(body: SearchRequest):
    try:
        return {"query": body.query, "hits": agent().search_jobs(body.query, body.top_k)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.post("/rag/ask")
def ask(body: AskRequest):
    try:
        return agent().ask_rag(body.question, body.top_k, body.use_llm)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
