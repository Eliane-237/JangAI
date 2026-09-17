"""
Routage de requete — detection generique des facettes.

Determine, a partir de la question, quels FILTRES appliquer a la recherche
(matiere, niveau, serie, et demain type de document, annee...). Rien n'est
code en dur : les valeurs candidates sont lues DANS la base (valeurs distinctes
des colonnes de facette). Ajouter un corpus d'epreuves ou de documents
administratifs ne demandera donc aucun changement ici — juste des documents
portant ces metadonnees.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from loguru import logger

from app.db import get_connection

# Colonnes de facette sur lesquelles router. Extensible : ajouter par exemple
# "document_type" ou "program_year" ici (une fois la colonne presente) suffit.
FACET_COLUMNS: tuple[str, ...] = ("subject", "level", "track")


def _strip(text: str) -> str:
    """Minuscule, sans accents."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(c)
    )


@lru_cache(maxsize=1)
def _facet_values() -> dict[str, set[str]]:
    """Valeurs distinctes de chaque facette, telles que presentes en base.

    Mise en cache pour la duree du processus. Les noms de colonnes viennent
    d'une constante (jamais de l'utilisateur) : pas d'injection SQL.
    """
    values: dict[str, set[str]] = {}
    try:
        with get_connection() as connection:
            for column in FACET_COLUMNS:
                rows = connection.execute(
                    f"SELECT DISTINCT {column} FROM documents WHERE {column} IS NOT NULL"
                ).fetchall()
                values[column] = {r[0].lower() for r in rows if r[0]}
    except Exception as exc:  # pragma: no cover - base indisponible
        logger.warning("Facettes indisponibles ({}) : routage sans filtre.", exc)
    return values


def detect_filters(question: str) -> dict[str, str]:
    """Detecte les facettes nommees dans la question.

    Une facette est retenue si l'une de ses valeurs connues apparait comme mot
    entier dans la question (insensible aux accents/casse). Exemple : « ... en
    français » -> {"subject": "francais"}.

    Args:
        question: Question de l'utilisateur

    Returns:
        Filtres a appliquer a la recherche (vide si aucune facette nommee)
    """
    normalized = _strip(question)
    filters: dict[str, str] = {}
    for column, candidates in _facet_values().items():
        for value in candidates:
            if value and re.search(rf"\b{re.escape(value)}\b", normalized):
                filters[column] = value
                break
    if filters:
        logger.info("Routage : facettes detectees {}", filters)
    return filters
