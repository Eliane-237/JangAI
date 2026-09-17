"""
API JangAI — point d'entree.

Assemble les routeurs du dossier `app/routes` derriere une application FastAPI.

    GET  /health   etat du service et de la base
    POST /search   candidats apres recherche + reranking (debogage, sans LLM)
    POST /ask      reponse complete generee, avec sources (pipeline lineaire)
    POST /chat     reponse via l'agent LangGraph (routage + recherche + gen.)

Lancement (usage normal, modeles gardes chauds) :
    uvicorn app.main:app

N'utiliser --reload QUE pendant le developpement : il redemarre le serveur a
chaque sauvegarde de fichier, ce qui recharge tous les modeles.
"""

from __future__ import annotations

from fastapi import FastAPI
from loguru import logger

from app.config import get_settings
from app.db import check_connection, get_statistics
from app.routes import chat, query, search

app = FastAPI(
    title="JangAI",
    description="Assistant de recherche sur les programmes scolaires senegalais",
    version="1.0.0",
)

app.include_router(query.router)
app.include_router(search.router)
app.include_router(chat.router)


@app.on_event("startup")
def _log_database_state() -> None:
    """Annonce au demarrage ce que le serveur voit reellement en base.

    Rend immediatement visible le cas ou le serveur pointe sur une base vide ou
    reinitialisee (pool accroche a une ancienne instance) : il suffit alors de
    relancer le serveur.
    """
    if not check_connection():
        logger.error("Base indisponible au demarrage : lancez `docker compose up -d`.")
        return

    from app.db import get_connection

    with get_connection() as c:
        target = c.execute(
            "SELECT inet_server_addr(), inet_server_port(), current_database(), "
            "(SELECT count(*) FROM pg_indexes WHERE tablename='chunks' "
            "AND indexname LIKE '%hnsw%')"
        ).fetchone()
    logger.info(
        "Cible Postgres : {}:{} / base {} | index HNSW: {}",
        target[0], target[1], target[2], "present" if target[3] else "ABSENT",
    )

    stats = get_statistics()
    logger.info(
        "Base vue par l'API : {} documents, {} chunks, {} vectorises",
        stats["documents"], stats["chunks"], stats["embedded"],
    )
    if not stats["chunks"]:
        logger.warning(
            "La base est vide cote API. Si vous venez de (re)creer le conteneur "
            "Postgres, relancez ce serveur pour rafraichir le pool de connexions."
        )


@app.on_event("startup")
def _warmup_models() -> None:
    """Precharge les modeles lourds AVANT la premiere requete.

    Sans cela, l'embedding et le reranker se chargent paresseusement a la
    premiere question, qui paie alors ~10 s d'attente. En les chargeant au
    demarrage, chaque requete demarre immediatement.
    """
    if not get_settings().warmup_models:
        return
    import time

    started = time.perf_counter()
    try:
        from app.retrieval.reranker import get_cross_encoder
        from pipeline.embedding import get_embedding_service

        get_embedding_service().embed_query("prechauffage")
        reranker = get_cross_encoder()
        if reranker is not None:
            reranker.score("prechauffage", ["prechauffage"])
        logger.info(
            "Modeles prechauffes en {:.1f}s (embedding + reranker) : "
            "requetes immediates.",
            time.perf_counter() - started,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Prechauffage des modeles incomplet ({}) : ils se chargeront a la "
            "premiere requete.", exc,
        )


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
