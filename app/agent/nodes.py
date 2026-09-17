"""
Noeuds de l'agent LangGraph.

Chaque noeud est une fonction `(state) -> mise a jour de l'etat`. Ils sont
volontairement MINCES : ils ne font que brancher les briques deja eprouvees de
JangAI (routage, recherche hybride + rerank, generation). LangGraph orchestre
l'enchainement ; la logique metier reste dans `app.services` / `app.retrieval`.
"""

from __future__ import annotations

from loguru import logger

from app.agent.state import AgentState
from app.services.generator import generate_answer
from app.services.query_router import detect_filters
from app.services.rag_pipeline import retrieve as pipeline_retrieve


def route_node(state: AgentState) -> dict:
    """Detecte les facettes (matiere, niveau...) nommees dans la question."""
    filters = detect_filters(state["question"])
    return {"filters": filters}


def retrieve_node(state: AgentState) -> dict:
    """Recherche hybride + reranking, filtree par les facettes detectees."""
    filters = state.get("filters") or None
    candidates = pipeline_retrieve(state["question"], filters=filters)
    logger.info("Agent : {} extraits retenus", len(candidates))
    return {"candidates": candidates}


def generate_node(state: AgentState) -> dict:
    """Genere la reponse citee a partir des extraits retenus."""
    answer = generate_answer(state["question"], state.get("candidates") or [])
    return {"answer": answer.text, "sources": [s.to_dict() for s in answer.sources]}
