"""
Reranking des candidats sur des signaux generiques.

La recherche hybride privilegie le rappel ; le reranking privilegie la
precision. Chaque candidat est note par une combinaison ponderee de quatre
signaux, tous independants du domaine (aucune liste de mots-cles metier) :

    vector_score     similarite semantique rendue par le vectoriel ;
    hierarchy_match  recouvrement des termes de la question avec le chemin de
                     titres du chunk ;
    term_density     part des termes de la question presents dans le texte ;
    reliability      confiance dans l'extraction (natif > OCR).

Les ponderations viennent de la configuration et totalisent 1,0.
"""

from __future__ import annotations

import re
import unicodedata

from app.config import get_settings
from app.retrieval.search import Candidate

_RELIABILITY_SCORE = {"high": 1.0, "medium": 0.6, "low": 0.3}


def _terms(text: str) -> set[str]:
    """Termes normalises d'un texte : minuscules, sans accent, longueur > 2."""
    stripped = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )
    return {w for w in re.findall(r"[a-z0-9]+", stripped.lower()) if len(w) > 2}


def _overlap(query_terms: set[str], text: str) -> float:
    """Part des termes de la question presents dans un texte (0 a 1)."""
    if not query_terms:
        return 0.0
    target = _terms(text)
    return len(query_terms & target) / len(query_terms)


def rerank(
    query: str, candidates: list[Candidate], top_k: int | None = None
) -> list[Candidate]:
    """Note et ordonne les candidats, puis retourne les meilleurs.

    Args:
        query: Question de l'utilisateur
        candidates: Candidats issus de la recherche hybride
        top_k: Nombre de candidats conserves (defaut : `rerank_top_k`)

    Returns:
        Candidats ordonnes par pertinence decroissante
    """
    settings = get_settings()
    top_k = top_k or settings.rerank_top_k
    query_terms = _terms(query)

    for candidate in candidates:
        hierarchy_match = _overlap(query_terms, candidate.hierarchy_text())
        term_density = _overlap(query_terms, candidate.content)
        reliability = _RELIABILITY_SCORE.get(candidate.reliability or "high", 1.0)

        candidate.rerank_score = (
            settings.weight_vector_score * candidate.vector_score
            + settings.weight_hierarchy_match * hierarchy_match
            + settings.weight_term_density * term_density
            + settings.weight_reliability * reliability
        )

    candidates.sort(key=lambda c: c.rerank_score, reverse=True)
    return candidates[:top_k]
