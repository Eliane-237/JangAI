"""
Recherche hybride : vectorielle (pgvector) + lexicale (plein-texte).

Deux voies complementaires, reunies avant le reranking :

    VECTORIELLE  similarite cosinus sur l'index HNSW. Retrouve le sens, y
                 compris quand les mots different (paraphrases, synonymes).
    LEXICALE     plein-texte francais (to_tsvector) sans accent, sur
                 `indexed_content` (titres compris). Rattrape les termes rares
                 et les codes exacts sur lesquels le vectoriel echoue.

L'union des candidats maximise le rappel ; la selection finale est laissee au
reranking (`app.retrieval.reranker`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from app.config import get_settings
from app.db import get_connection
from app.retrieval.filters import build_filter_clause, ts_or_query
from pipeline.embedding import get_embedding_service

# Colonnes lues pour chaque candidat, dans un ordre fixe.
_COLUMNS = (
    "chunk_id, content, source_file, subject, level, track, "
    "page_number, chunk_type, reliability, hierarchy"
)


@dataclass
class Candidate:
    """Un chunk candidat, avec ses scores de recherche."""

    chunk_id: str
    content: str
    source_file: str
    subject: str | None
    level: str | None
    track: str | None
    page_number: int | None
    chunk_type: str | None
    reliability: str | None
    hierarchy_path: list[str] = field(default_factory=list)
    vector_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0

    @classmethod
    def from_row(cls, row: tuple) -> "Candidate":
        (
            chunk_id, content, source_file, subject, level, track,
            page_number, chunk_type, reliability, hierarchy,
        ) = row
        return cls(
            chunk_id=chunk_id,
            content=content,
            source_file=source_file,
            subject=subject,
            level=level,
            track=track,
            page_number=page_number,
            chunk_type=chunk_type,
            reliability=reliability,
            hierarchy_path=list(hierarchy) if hierarchy else [],
        )

    def hierarchy_text(self) -> str:
        return " > ".join(self.hierarchy_path)


def vector_search(query_vector, top_k: int, filters=None) -> list[Candidate]:
    """Recherche par similarite cosinus sur l'index HNSW."""
    settings = get_settings()
    where_extra, extra_params = build_filter_clause(filters)

    sql = (
        f"SELECT {_COLUMNS}, 1 - (embedding <=> %s) AS vector_score "
        f"FROM chunks WHERE embedding IS NOT NULL{where_extra} "
        f"ORDER BY embedding <=> %s LIMIT %s"
    )
    with get_connection() as connection:
        connection.execute(f"SET hnsw.ef_search = {int(settings.hnsw_ef_search)}")
        rows = connection.execute(
            sql, (query_vector, *extra_params, query_vector, top_k)
        ).fetchall()

        # 0 ligne est anormal des que la base contient des vecteurs : le plus
        # souvent un filtre trop restrictif. On le signale avec les filtres en
        # cause pour un diagnostic immediat.
        if not rows:
            total = connection.execute(
                "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            logger.warning(
                "vector_search 0 resultat ({} vecteurs en base) — filtres appliques : {}",
                total, filters or "aucun",
            )

    candidates = []
    for row in rows:
        candidate = Candidate.from_row(row[:-1])
        candidate.vector_score = float(row[-1])
        candidates.append(candidate)
    return candidates


def lexical_search(query: str, top_k: int, filters=None) -> list[Candidate]:
    """Recherche plein-texte francaise, insensible aux accents."""
    ts_query = ts_or_query(query)
    if not ts_query:
        return []

    where_extra, extra_params = build_filter_clause(filters)

    # La recherche porte sur `indexed_content` (contexte hierarchique + texte),
    # non sur `content` : les termes qui vivent dans les titres — souvent les
    # plus discriminants — sont ainsi cherchables.
    sql = (
        f"SELECT {_COLUMNS}, "
        f"ts_rank_cd(to_tsvector('fr_unaccent', indexed_content), "
        f"to_tsquery('fr_unaccent', %s)) AS lexical_score "
        f"FROM chunks "
        f"WHERE to_tsvector('fr_unaccent', indexed_content) @@ "
        f"to_tsquery('fr_unaccent', %s){where_extra} "
        f"ORDER BY lexical_score DESC LIMIT %s"
    )
    with get_connection() as connection:
        rows = connection.execute(
            sql, (ts_query, ts_query, *extra_params, top_k)
        ).fetchall()

    candidates = []
    for row in rows:
        candidate = Candidate.from_row(row[:-1])
        candidate.lexical_score = float(row[-1])
        candidates.append(candidate)
    return candidates


def _backfill_vector_scores(query_vector, candidates: list[Candidate]) -> None:
    """Complete le score vectoriel des candidats issus du seul lexical."""
    missing = [c for c in candidates if c.vector_score == 0.0]
    if not missing:
        return
    ids = [c.chunk_id for c in missing]
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT chunk_id, 1 - (embedding <=> %s) FROM chunks "
            "WHERE chunk_id = ANY(%s) AND embedding IS NOT NULL",
            (query_vector, ids),
        ).fetchall()
    scores = {chunk_id: float(score) for chunk_id, score in rows}
    for candidate in missing:
        candidate.vector_score = scores.get(candidate.chunk_id, 0.0)


def hybrid_search(query: str, top_k: int | None = None, filters=None) -> list[Candidate]:
    """Union des candidats vectoriels et lexicaux, scores complets.

    Args:
        query: Question de l'utilisateur
        top_k: Nombre de candidats par voie (defaut : `search_top_k`)
        filters: Filtres optionnels (subject, level, track, ...)

    Returns:
        Candidats dedupliques, prets pour le reranking
    """
    settings = get_settings()
    top_k = top_k or settings.search_top_k

    query_vector = get_embedding_service().embed_query(query)
    vector_hits = vector_search(query_vector, top_k, filters)
    lexical_hits = lexical_search(query, top_k, filters)

    merged: dict[str, Candidate] = {}
    for candidate in vector_hits:
        merged[candidate.chunk_id] = candidate
    for candidate in lexical_hits:
        if candidate.chunk_id in merged:
            merged[candidate.chunk_id].lexical_score = candidate.lexical_score
        else:
            merged[candidate.chunk_id] = candidate

    candidates = list(merged.values())
    _backfill_vector_scores(query_vector, candidates)

    logger.info(
        "Recherche '{}' : {} vectoriels + {} lexicaux -> {} candidats",
        query[:50], len(vector_hits), len(lexical_hits), len(candidates),
    )
    return candidates
