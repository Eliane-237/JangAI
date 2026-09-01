"""
Filtres et preparation de requete pour la recherche.

Deux helpers transverses, partages par la recherche vectorielle et lexicale :
la construction d'une clause WHERE sur les colonnes de filtrage typees, et la
transformation d'une question en `tsquery` OU pour le plein-texte.
"""

from __future__ import annotations

import re
import unicodedata

# Seules ces colonnes typees et indexees peuvent servir de filtre.
_ALLOWED_FILTERS = {"subject", "level", "track", "cycle", "language"}


def build_filter_clause(filters: dict[str, str] | None) -> tuple[str, list]:
    """Construit une clause WHERE optionnelle sur les colonnes de filtrage.

    Args:
        filters: Couples colonne -> valeur (subject, level, track, ...)

    Returns:
        Tuple (fragment SQL prefixe de " AND ", parametres correspondants)
    """
    if not filters:
        return "", []
    clauses, params = [], []
    for key, value in filters.items():
        if key in _ALLOWED_FILTERS and value:
            # Comparaison insensible aux accents ET a la casse : un filtre
            # "Francais", "français" ou "FRANCAIS" retrouve la valeur stockee
            # "francais", et "ls"/"LS" se rejoignent. Sans quoi un simple
            # accent cote client renverrait zero resultat.
            clauses.append(f"unaccent(lower({key})) = unaccent(lower(%s))")
            params.append(value)
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def ts_or_query(query: str) -> str:
    """Transforme une question en `tsquery` OU sur ses termes significatifs.

    `plainto_tsquery` relie les termes par ET — un chunk devrait alors
    contenir TOUS les mots. Pour du rappel, on relie par OU (`|`) les seuls
    termes porteurs (longueur > 2), apres suppression des accents.

    Args:
        query: Question brute

    Returns:
        Chaine `tsquery` (ex. "objectifs | lecture | francais"), vide si aucun
        terme exploitable
    """
    stripped = "".join(
        c for c in unicodedata.normalize("NFKD", query) if not unicodedata.combining(c)
    )
    terms = [w for w in re.findall(r"[a-z0-9]+", stripped.lower()) if len(w) > 2]
    return " | ".join(dict.fromkeys(terms))
