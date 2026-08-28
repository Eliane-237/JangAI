"""
Pipeline RAG — orchestration de la chaine de reponse.

    recherche hybride  ->  reranking  ->  generation

Point d'entree unique appele par les routes. Les etages restent independants
et testables separement ; ce module ne fait que les enchainer. C'est aussi
l'endroit ou brancheront, plus tard, les etages avances du dossier
`app/services` (routage de requete, decomposition, cache semantique, garde-fous).
"""

from __future__ import annotations

from loguru import logger

from app.retrieval.reranker import rerank
from app.retrieval.search import Candidate, hybrid_search
from app.services.generator import Answer, generate_answer


def retrieve(query: str, top_k: int | None = None, filters=None) -> list[Candidate]:
    """Recherche + reranking, sans generation (utile pour le debogage)."""
    candidates = hybrid_search(query, filters=filters)
    return rerank(query, candidates, top_k=top_k)


def answer_question(query: str, top_k: int | None = None, filters=None) -> Answer:
    """Repond a une question de bout en bout.

    Args:
        query: Question de l'utilisateur
        top_k: Nombre d'extraits transmis au LLM (defaut : `rerank_top_k`)
        filters: Filtres optionnels (subject, level, track, ...)

    Returns:
        La reponse generee et ses sources
    """
    logger.info("Question : {}", query)
    candidates = retrieve(query, top_k=top_k, filters=filters)
    if not candidates:
        return Answer(
            text="Aucun document pertinent trouve. La base est-elle bien ingeree ?",
            sources=[],
        )
    return generate_answer(query, candidates)
