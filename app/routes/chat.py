"""
Route /chat — reponse via l'agent LangGraph.

Contrairement a /ask (pipeline lineaire fige), /chat passe par le GRAPHE de
l'agent : routage des facettes -> recherche -> generation (et, en phase 2, la
boucle auto-corrective). L'agent decide lui-meme des filtres a appliquer.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.agent.graph import answer as run_agent
from app.routes.schemas import ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Repond a une question via l'agent (routage + recherche + generation)."""
    try:
        return run_agent(request.question)
    except RuntimeError as exc:
        # Cle Groq absente : configuration incomplete, pas une erreur serveur.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Echec de l'agent")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
