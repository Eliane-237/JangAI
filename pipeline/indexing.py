"""
Ecriture en base et indexation vectorielle.

Insere les documents et leurs vecteurs dans PostgreSQL, puis construit
l'index HNSW.

Deux choix expliques ici :

    - L'insertion est IDEMPOTENTE. Les identifiants de chunk sont
      deterministes, et un `ON CONFLICT DO UPDATE` remplace la ligne
      existante. Reingerer deux fois le meme fichier ne cree aucun doublon.

    - L'index HNSW est reconstruit APRES insertion. Construire un graphe
      sur une table remplie est plus rapide et produit un graphe de
      meilleure qualite qu'une construction incrementale.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from loguru import logger

from app.config import get_settings
from app.db import get_connection
from app.models import Document

# ======================================================================
# Documents source
# ======================================================================


def save_document(
    document_id: str,
    source_file: str,
    identity,
    page_count: int,
    gaps: list[Any],
    diagnostics: dict[str, Any],
    sub_document_id: str | None = None,
) -> None:
    """Cree ou met a jour la fiche d'un document source.

    Args:
        document_id: Identifiant du document
        source_file: Nom du fichier
        identity: Identite deduite (`DocumentIdentity`)
        page_count: Nombre de pages du fichier
        gaps: Pages internes manquantes
        diagnostics: Rapport d'analyse structurelle
        sub_document_id: Sous-document concerne
    """
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                document_id, source_file, sub_document_id,
                subject, level, track, cycle, language, program_year,
                document_profile, page_count, gaps, diagnostics
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (document_id) DO UPDATE SET
                subject          = EXCLUDED.subject,
                level            = EXCLUDED.level,
                track            = EXCLUDED.track,
                cycle            = EXCLUDED.cycle,
                language         = EXCLUDED.language,
                document_profile = EXCLUDED.document_profile,
                page_count       = EXCLUDED.page_count,
                gaps             = EXCLUDED.gaps,
                diagnostics      = EXCLUDED.diagnostics,
                updated_at       = now()
            """,
            (
                document_id, source_file, sub_document_id,
                identity.subject, identity.level, identity.track,
                identity.cycle, identity.language, identity.program_year,
                identity.profile.value, page_count,
                json.dumps(gaps, ensure_ascii=False),
                json.dumps(diagnostics, ensure_ascii=False),
            ),
        )


# ======================================================================
# Chunks
# ======================================================================

_INSERT_CHUNK = """
INSERT INTO chunks (
    chunk_id, document_id, source_file,
    content, indexed_content, content_hash,
    subject, level, track, cycle, language, document_profile,
    page_number, printed_page, sub_document_id, chunk_order,
    chunk_type, page_layout, extraction_method, reliability, char_count,
    hierarchy, metadata, embedding
) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (chunk_id) DO UPDATE SET
    content         = EXCLUDED.content,
    indexed_content = EXCLUDED.indexed_content,
    content_hash    = EXCLUDED.content_hash,
    hierarchy       = EXCLUDED.hierarchy,
    metadata        = EXCLUDED.metadata,
    reliability     = EXCLUDED.reliability,
    embedding       = EXCLUDED.embedding
"""


def _to_row(document: Document, vector: np.ndarray | None) -> tuple:
    """Prepare le tuple de parametres d'un chunk."""
    meta = document.metadata
    pos = meta.position
    return (
        meta.chunk_id,
        meta.document_id,
        meta.source_file,
        document.page_content,
        document.contextualized_text(),
        meta.content_hash,
        meta.subject,
        meta.level,
        meta.track,
        meta.cycle,
        meta.language,
        meta.document_profile.value,
        pos.page_number if pos else None,
        pos.printed_page if pos else None,
        pos.sub_document_id if pos else None,
        pos.order if pos else None,
        meta.chunk_type.value,
        meta.page_layout.value,
        meta.extraction_method.value,
        meta.reliability.value,
        meta.char_count,
        json.dumps(meta.hierarchy.path(), ensure_ascii=False),
        json.dumps(
            {
                "merged_across_pages": meta.merged_across_pages,
                "program_year": meta.program_year,
                "row_index": pos.row_index if pos else None,
            },
            ensure_ascii=False,
        ),
        None if vector is None else np.asarray(vector, dtype=np.float32),
    )


def insert_documents_with_metadata(
    documents: list[Document], embeddings: np.ndarray, batch_size: int = 200
) -> int:
    """Insere les chunks et leurs vecteurs, par lots.

    Le decoupage en lots evite de construire une requete unique de
    plusieurs dizaines de megaoctets sur un corpus important.

    Args:
        documents: Chunks a inserer
        embeddings: Vecteurs correspondants, terme a terme
        batch_size: Nombre de chunks par lot

    Returns:
        Nombre de chunks inseres

    Raises:
        ValueError: Si les deux listes n'ont pas la meme longueur
    """
    if not documents:
        return 0
    if len(documents) != len(embeddings):
        raise ValueError(
            f"{len(documents)} documents pour {len(embeddings)} vecteurs : "
            f"les deux listes doivent correspondre terme a terme."
        )

    total = 0
    with get_connection() as connection:
        with connection.cursor() as cursor:
            for start in range(0, len(documents), batch_size):
                batch = documents[start : start + batch_size]
                vectors = embeddings[start : start + batch_size]
                cursor.executemany(
                    _INSERT_CHUNK, [_to_row(d, v) for d, v in zip(batch, vectors)]
                )
                total += len(batch)
                logger.debug("{}/{} chunks inseres", total, len(documents))

    # Le compteur de la fiche document est recalcule depuis les faits.
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE documents d SET chunk_count = (
                SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.document_id
            )
            """
        )
    logger.info("{} chunks inseres", total)
    return total


# ======================================================================
# Index vectoriel
# ======================================================================


def rebuild_index() -> None:
    """Reconstruit l'index HNSW sur la table remplie.

    A appeler apres une ingestion complete. La reconstruction en bloc est
    nettement plus rapide qu'une insertion index actif, et produit un
    graphe de navigation de meilleure qualite.
    """
    settings = get_settings()
    logger.info(
        "Reconstruction de l'index HNSW (m={}, ef_construction={})",
        settings.hnsw_m, settings.hnsw_ef_construction,
    )
    with get_connection() as connection:
        connection.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
        connection.execute(
            f"""
            CREATE INDEX idx_chunks_embedding_hnsw
                ON chunks USING hnsw (embedding {settings.hnsw_operator})
                WITH (m = {settings.hnsw_m},
                      ef_construction = {settings.hnsw_ef_construction})
            """
        )
        connection.execute("ANALYZE chunks")
    logger.info("Index HNSW reconstruit")


def delete_document(document_id: str) -> int:
    """Supprime un document et ses chunks (cascade).

    Args:
        document_id: Identifiant du document

    Returns:
        Nombre de lignes supprimees
    """
    with get_connection() as connection:
        result = connection.execute(
            "DELETE FROM documents WHERE document_id = %s", (document_id,)
        )
        return result.rowcount


def get_quality_report() -> list[dict[str, Any]]:
    """Lit la vue de controle qualite construite par le schema.

    Returns:
        Une ligne par document source
    """
    with get_connection() as connection:
        cursor = connection.execute("SELECT * FROM v_ingestion_quality")
        columns = [d[0] for d in cursor.description or []]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]