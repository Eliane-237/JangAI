"""
API JangAI — point d'entree.

Assemble les routeurs du dossier `app/routes` derriere une application FastAPI.

    GET  /health   etat du service et de la base
    POST /search   candidats apres recherche + reranking (debogage, sans LLM)
    POST /ask      reponse complete generee, avec sources

Lancement :
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from app.config import get_settings
from app.db import check_connection, get_statistics
from app.routes import query, search

app = FastAPI(
    title="JangAI",
    description="Assistant de recherche sur les programmes scolaires senegalais",
    version="1.0.0",
)

app.include_router(query.router)
app.include_router(search.router)


@app.get("/health", tags=["monitoring"])
def health() -> dict:
    """Etat du service, de la base et de la generation."""
    connected = check_connection()
    stats = get_statistics() if connected else {}
    return {
        "status": "ok" if connected else "degraded",
        "database": connected,
        "documents": stats.get("documents", 0),
        "chunks": stats.get("chunks", 0),
        "embedded": stats.get("embedded", 0),
        "generation": bool(get_settings().groq_api_key),
    }
