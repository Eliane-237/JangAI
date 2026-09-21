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


# Synonymes de matiere : le LLM (ou l'utilisateur) ecrit « mathematiques »,
# la base stocke « maths ». On rabat ces variantes sur une forme pivot avant
# de la confronter aux valeurs reelles de la base. Cle et valeur SANS accents.
_SUBJECT_SYNONYMS = {
    "mathematiques": "maths",
    "mathematique": "maths",
    "math": "maths",
    "maths": "maths",
    "francais": "francais",
    "lettres": "francais",
    "philo": "philosophie",
}


def _common_prefix_len(a: str, b: str) -> int:
    """Longueur du prefixe commun a deux chaines."""
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _match_db_value(column: str, value: str, db_values: set[str]) -> str | None:
    """Rabat une valeur demandee sur la valeur REELLE de la base, ou None.

    Strategie, du plus sur au plus tolerant : synonyme connu, egalite,
    inclusion (prefixe dans un sens ou l'autre : « maths »/« mathematiques »),
    puis prefixe commun d'au moins 4 lettres. Aucun rapprochement -> None :
    l'appelant laissera alors tomber le filtre plutot que de vider la recherche.
    """
    v = _strip(value)
    if column == "subject":
        v = _SUBJECT_SYNONYMS.get(v, v)
    stripped = {d: _strip(d) for d in db_values}
    for original, d in stripped.items():
        if d == v:
            return original
    for original, d in stripped.items():
        if d.startswith(v) or v.startswith(d):
            return original
    for original, d in stripped.items():
        if len(v) >= 4 and len(d) >= 4 and _common_prefix_len(d, v) >= 4:
            return original
    return None


def canonicalize_filters(filters: dict[str, str] | None) -> dict[str, str]:
    """Aligne des filtres demandes sur les valeurs reelles des facettes en base.

    Indispensable cote agent : le LLM ecrit « mathematiques » / « terminale L »,
    la base stocke « maths » / « terminale ». Sans cet alignement, le filtre
    exact ne rapporte RIEN. Une facette qui ne correspond a aucune valeur connue
    est retiree (recherche elargie), jamais conservee telle quelle.

    Args:
        filters: Filtres proposes (par l'agent ou le client)

    Returns:
        Filtres alignes sur la base ; les facettes sans correspondance sont omises
    """
    if not filters:
        return {}
    facets = _facet_values()
    aligned: dict[str, str] = {}
    for column, value in filters.items():
        if not value:
            continue
        db_values = facets.get(column)
        if not db_values:
            logger.info("Filtre {}='{}' : facette inconnue -> ignore.", column, value)
            continue
        match = _match_db_value(column, value, db_values)
        if match:
            if _strip(match) != _strip(value):
                logger.info("Filtre {} : '{}' -> '{}' (valeur en base).", column, value, match)
            aligned[column] = match
        else:
            logger.info("Filtre {}='{}' sans equivalent en base -> ignore.", column, value)
    return aligned


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
