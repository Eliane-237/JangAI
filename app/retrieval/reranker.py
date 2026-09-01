"""
Reranking des candidats.

Deux strategies, la premiere nettement plus fine :

    CROSS-ENCODER (defaut)  Qwen3-Reranker juge la pertinence de chaque paire
                            (question, chunk) en lisant les deux ensemble. Il
                            capte le SENS, pas la simple presence de mots : un
                            chunk de maths ne remonte pas sur une question de
                            francais juste parce qu'il contient "eleve" ou
                            "competences".

    HEURISTIQUE (repli)     combinaison ponderee de signaux generiques (score
                            vectoriel, recouvrement de titres, densite de
                            termes, fiabilite). Utilisee si le cross-encoder est
                            indisponible ou desactive.

Le cross-encoder etant couteux sur CPU, seuls les meilleurs candidats (par
score vectoriel) sont reranked, dans la limite de `rerank_max_candidates`.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from loguru import logger

from app.config import get_settings
from app.db import get_connection
from app.retrieval.search import Candidate

_RELIABILITY_SCORE = {"high": 1.0, "medium": 0.6, "low": 0.3}

# Facteur applique a un chunk dont la matiere ne correspond pas a celle nommee
# dans la question. Ni 0 (on garde un repli si rien d'autre ne matche) ni 1
# (sinon la fuite inter-matiere persiste).
_SUBJECT_MISMATCH_FACTOR = 0.25


@lru_cache(maxsize=1)
def _db_subjects() -> tuple[str, ...]:
    """Matieres reellement presentes en base (normalisees), pour le routage.

    Data-driven : aucune liste de matieres codee en dur. Le cache est valable
    pour la duree du processus.
    """
    try:
        with get_connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT subject FROM documents WHERE subject IS NOT NULL"
            ).fetchall()
        return tuple(r[0].lower() for r in rows if r[0])
    except Exception:  # pragma: no cover - base indisponible
        return ()


def _subject_gate(query: str, candidates: list[Candidate]) -> list[float] | None:
    """Facteur de matiere par candidat, ou None si la question ne nomme aucune
    matiere connue.

    Si la question mentionne une matiere presente en base, les chunks d'une
    AUTRE matiere sont fortement penalises : c'est ce qui empeche un chapitre
    de maths de remonter sur une question de francais.
    """
    subjects = _db_subjects()
    if not subjects:
        return None
    normalized = "".join(
        c for c in unicodedata.normalize("NFKD", query.lower())
        if not unicodedata.combining(c)
    )
    mentioned = {s for s in subjects if s and s in normalized}
    if not mentioned:
        return None
    return [
        1.0 if (c.subject or "").lower() in mentioned else _SUBJECT_MISMATCH_FACTOR
        for c in candidates
    ]

# Gabarit officiel Qwen3-Reranker : le modele repond "yes"/"no" a la question
# "le document repond-il a la requete ?". Le score est la probabilite de "yes".
_PROMPT = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based "
    "on the Query. Answer only yes or no.<|im_end|>\n<|im_start|>user\n"
    "<Query>: {query}\n<Document>: {doc}<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)


# ======================================================================
# Cross-encoder Qwen3-Reranker
# ======================================================================


class CrossEncoderReranker:
    """Reranker par cross-encoder Qwen3-Reranker, charge paresseusement."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._tokenizer = None
        self._model = None
        self._yes_id = None
        self._no_id = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = self._settings.reranker_model
        logger.info("Chargement du reranker {} sur {}", name, self._settings.reranker_device)
        self._tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
        self._model = (
            AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
            .to(self._settings.reranker_device)
            .eval()
        )
        self._yes_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._no_id = self._tokenizer.convert_tokens_to_ids("no")
        logger.info("Reranker charge")

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score de pertinence (0 a 1) de chaque document pour la question.

        Args:
            query: Question de l'utilisateur
            documents: Contenus des chunks candidats

        Returns:
            Un score par document, dans l'ordre d'entree
        """
        import torch
        import torch.nn.functional as F

        if self._model is None:
            self._load()

        settings = self._settings
        prompts = [_PROMPT.format(query=query, doc=doc) for doc in documents]
        scores: list[float] = []

        for start in range(0, len(prompts), settings.reranker_batch_size):
            batch = prompts[start : start + settings.reranker_batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.reranker_max_length,
            ).to(settings.reranker_device)

            with torch.no_grad():
                logits = self._model(**inputs).logits[:, -1, :]  # dernier token
            pair = torch.stack([logits[:, self._no_id], logits[:, self._yes_id]], dim=1)
            yes_prob = F.softmax(pair, dim=1)[:, 1]
            scores.extend(yes_prob.tolist())

        return scores


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoderReranker | None:
    """Instance unique du cross-encoder, ou None s'il est indisponible."""
    settings = get_settings()
    if not settings.use_cross_encoder:
        return None
    try:
        return CrossEncoderReranker()
    except Exception as exc:  # pragma: no cover
        logger.warning("Cross-encoder indisponible ({}) : repli heuristique.", exc)
        return None


# ======================================================================
# Reranking heuristique (repli)
# ======================================================================


def _terms(text: str) -> set[str]:
    """Termes normalises d'un texte : minuscules, sans accent, longueur > 2."""
    stripped = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )
    return {w for w in re.findall(r"[a-z0-9]+", stripped.lower()) if len(w) > 2}


def _overlap(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & _terms(text)) / len(query_terms)


def _normalize(values: list[float]) -> list[float]:
    """Ramene une liste de scores dans [0,1] (min-max) pour les rendre
    comparables avant fusion. Constante -> 1.0, comme dans le reranker de
    reference."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [1.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def _rerank_heuristic(
    query: str, candidates: list[Candidate], top_k: int
) -> list[Candidate]:
    """Note les candidats par combinaison ponderee de signaux generiques."""
    settings = get_settings()
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


# ======================================================================
# Point d'entree
# ======================================================================


def rerank(
    query: str, candidates: list[Candidate], top_k: int | None = None
) -> list[Candidate]:
    """Ordonne les candidats par pertinence et retourne les meilleurs.

    Args:
        query: Question de l'utilisateur
        candidates: Candidats issus de la recherche hybride
        top_k: Nombre de candidats conserves (defaut : `rerank_top_k`)

    Returns:
        Candidats ordonnes par pertinence decroissante
    """
    settings = get_settings()
    top_k = top_k or settings.rerank_top_k
    if not candidates:
        return []

    reranker = get_cross_encoder()
    if reranker is None:
        return _rerank_heuristic(query, candidates, top_k)

    # Le cross-encoder est couteux : on le concentre sur les meilleurs candidats
    # par score vectoriel, sans depasser rerank_max_candidates.
    shortlist = sorted(candidates, key=lambda c: c.vector_score, reverse=True)[
        : settings.rerank_max_candidates
    ]
    query_terms = _terms(query)
    try:
        # On donne au reranker la matiere et la section en plus du texte : il
        # peut ainsi ecarter un chunk hors-sujet (maths sur une question de
        # francais) que le seul contenu ferait paraitre pertinent.
        docs = [
            f"{c.subject or ''} {c.hierarchy_text()}\n{c.content}".strip()
            for c in shortlist
        ]
        cross = reranker.score(query, docs)
    except Exception as exc:  # pragma: no cover
        logger.warning("Echec du cross-encoder ({}) : repli heuristique.", exc)
        return _rerank_heuristic(query, candidates, top_k)

    # Fusion multi-signaux : le cross-encoder porte le sens, mais un chunk doit
    # aussi etre ancre lexicalement et vectoriellement. Chaque signal est
    # normalise sur les candidats avant ponderation. C'est ce qui retrograde un
    # chunk hors-matiere que le seul cross-encoder trouverait pertinent.
    cross_n = _normalize(cross)
    vector_n = _normalize([c.vector_score for c in shortlist])
    lexical_n = _normalize([c.lexical_score for c in shortlist])
    term_n = _normalize([_overlap(query_terms, c.content) for c in shortlist])
    # Porte de matiere : penalise le hors-matiere quand la question la nomme.
    subject_gate = _subject_gate(query, shortlist)

    for i, candidate in enumerate(shortlist):
        score = (
            settings.rerank_w_cross * cross_n[i]
            + settings.rerank_w_vector * vector_n[i]
            + settings.rerank_w_lexical * lexical_n[i]
            + settings.rerank_w_term * term_n[i]
        )
        if subject_gate is not None:
            score *= subject_gate[i]
        candidate.rerank_score = score

    ranked = sorted(shortlist, key=lambda c: c.rerank_score, reverse=True)
    best = ranked[0] if ranked else None
    logger.info(
        "Rerank fusion : {} candidats -> {} retenus | meilleur {:.3f} "
        "(cross-encoder brut {:.3f})",
        len(shortlist), min(top_k, len(ranked)),
        best.rerank_score if best else 0.0,
        max(cross) if cross else 0.0,
    )
    return ranked[:top_k]
