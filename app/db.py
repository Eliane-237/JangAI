"""
Acces PostgreSQL.

Pool de connexions psycopg3 partage, avec enregistrement du type `vector`
de pgvector. Toutes les couches superieures passent par ce module ; aucune
n'ouvre de connexion directe.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

import psycopg
from loguru import logger
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import get_settings


@lru_cache(maxsize=1)
def get_pool() -> ConnectionPool:
    """Pool de connexions unique pour tout le processus.

    `configure` est appele sur chaque nouvelle connexion : c'est la que le
    type `vector` est enregistre, sans quoi psycopg ne saurait pas
    convertir une liste Python en vecteur pgvector.

    Returns:
        Le pool de connexions
    """
    settings = get_settings()

    def configure(connection: psycopg.Connection) -> None:
        register_vector(connection)

    pool = ConnectionPool(
        conninfo=settings.dsn,
        min_size=1,
        max_size=8,
        configure=configure,
        open=True,
        timeout=30,
    )
    logger.info(
        "Pool PostgreSQL ouvert sur {}:{}/{}",
        settings.postgres_host, settings.postgres_port, settings.postgres_db,
    )
    return pool


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Connexion empruntee au pool, validee ou annulee automatiquement."""
    with get_pool().connection() as connection:
        yield connection


def execute(query: str, params: tuple[Any, ...] | None = None) -> None:
    """Execute une requete sans resultat attendu."""
    with get_connection() as connection:
        connection.execute(query, params)


def fetch_all(query: str, params: tuple[Any, ...] | None = None) -> list[tuple]:
    """Execute une requete et retourne toutes les lignes."""
    with get_connection() as connection:
        return connection.execute(query, params).fetchall()


def check_connection() -> bool:
    """Controle la disponibilite de la base et des extensions requises.

    Returns:
        True si la base est prete a recevoir des donnees
    """
    try:
        with get_connection() as connection:
            version = connection.execute("SELECT version()").fetchone()
            extensions = connection.execute(
                "SELECT extname FROM pg_extension "
                "WHERE extname IN ('vector', 'pg_trgm', 'unaccent')"
            ).fetchall()
        present = {e[0] for e in extensions}
        logger.info("PostgreSQL : {}", (version[0] if version else "?")[:60])
        logger.info("Extensions : {}", ", ".join(sorted(present)) or "aucune")
        if "vector" not in present:
            logger.error("Extension pgvector absente : schema non initialise.")
            return False
        return True
    except Exception as exc:
        logger.error("Connexion PostgreSQL impossible : {}", exc)
        return False


def get_statistics() -> dict[str, Any]:
    """Etat courant de la base, pour les controles apres ingestion.

    Returns:
        Compteurs de documents, de chunks et repartition par matiere
    """
    with get_connection() as connection:
        documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()
        chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
        embedded = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()
        by_subject = connection.execute(
            "SELECT subject, COUNT(*) FROM chunks GROUP BY subject ORDER BY 2 DESC"
        ).fetchall()
    return {
        "documents": documents[0] if documents else 0,
        "chunks": chunks[0] if chunks else 0,
        "embedded": embedded[0] if embedded else 0,
        "by_subject": dict(by_subject),
    }