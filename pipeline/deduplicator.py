"""
Deduplication des chunks.

Trois niveaux de detection, du moins couteux au plus couteux :

    1. EXACT      empreinte du contenu normalise. Elimine les repetitions
                  strictes : en-tetes recurrents, pages dupliquees dans un
                  fichier assemble.
    2. CONTENU    empreinte identique a la casse et aux espaces pres.
    3. APPROCHE   similarite de Levenshtein via rapidfuzz, pour les quasi
                  doublons produits par l'OCR ou les recollements.

La comparaison approchee n'est appliquee qu'a l'interieur d'une meme
matiere : comparer tous les chunks entre eux serait quadratique et sans
objet, deux matieres n'ayant pas a partager de contenu.
"""

from __future__ import annotations

from collections import defaultdict

from loguru import logger

from app.config import THRESHOLDS
from app.models import Document, Reliability

try:
    from rapidfuzz import fuzz

    FUZZY_AVAILABLE = True
except ImportError:  # pragma: no cover
    FUZZY_AVAILABLE = False
    logger.warning("rapidfuzz indisponible : deduplication approchee desactivee.")


# Ordre de preference lorsqu'il faut choisir entre deux doublons.
_RELIABILITY_RANK = {
    Reliability.HIGH: 3,
    Reliability.MEDIUM: 2,
    Reliability.LOW: 1,
}


def _preference_score(document: Document) -> tuple[int, int]:
    """Score de preference entre deux doublons.

    On conserve le chunk le plus fiable ; a fiabilite egale, le plus long,
    qui porte generalement le contenu le plus complet.

    Args:
        document: Chunk candidat

    Returns:
        Tuple de tri (rang de fiabilite, longueur)
    """
    return (
        _RELIABILITY_RANK.get(document.metadata.reliability, 0),
        document.metadata.char_count,
    )


def remove_exact_duplicates(documents: list[Document]) -> list[Document]:
    """Supprime les chunks dont le contenu normalise est identique.

    Args:
        documents: Chunks a filtrer

    Returns:
        Chunks sans doublon exact
    """
    best_by_hash: dict[str, Document] = {}
    order: list[str] = []

    for document in documents:
        key = document.metadata.content_hash or ""
        if not key:
            continue
        if key not in best_by_hash:
            best_by_hash[key] = document
            order.append(key)
        elif _preference_score(document) > _preference_score(best_by_hash[key]):
            best_by_hash[key] = document

    removed = len(documents) - len(order)
    if removed:
        logger.info("{} doublon(s) exact(s) supprime(s)", removed)
    return [best_by_hash[key] for key in order]


def find_near_duplicates(
    documents: list[Document], threshold: int
) -> set[str]:
    """Identifie les chunks quasi identiques a l'interieur d'une matiere.

    Args:
        documents: Chunks a comparer
        threshold: Similarite (0-100) au-dela de laquelle deux chunks sont
            tenus pour doublons

    Returns:
        Ensemble des identifiants de chunks a ecarter
    """
    if not FUZZY_AVAILABLE:
        return set()

    by_subject: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        by_subject[document.metadata.subject or "?"].append(document)

    to_drop: set[str] = set()
    for subject, group in by_subject.items():
        # Le tri par longueur permet d'ecarter tot les paires dont les
        # tailles sont trop eloignees pour etre similaires.
        group = sorted(group, key=lambda d: d.metadata.char_count)
        for index, candidate in enumerate(group):
            if candidate.metadata.chunk_id in to_drop:
                continue
            for other in group[index + 1 :]:
                if other.metadata.chunk_id in to_drop:
                    continue
                shorter, longer = candidate.metadata.char_count, other.metadata.char_count
                if longer and shorter / longer < threshold / 100:
                    break  # trop court pour atteindre le seuil
                score = fuzz.ratio(candidate.page_content, other.page_content)
                if score >= threshold:
                    weaker = min(candidate, other, key=_preference_score)
                    to_drop.add(weaker.metadata.chunk_id)
                    logger.debug(
                        "Quasi-doublon ({}%) ecarte dans '{}'", int(score), subject
                    )
    return to_drop


def deduplicate(
    documents: list[Document], fuzzy: bool = True
) -> list[Document]:
    """Supprime les doublons exacts puis, si demande, les quasi-doublons.

    Args:
        documents: Chunks a nettoyer
        fuzzy: Active la comparaison approchee

    Returns:
        Chunks dedupliques
    """
    if not documents:
        return documents

    initial = len(documents)
    result = remove_exact_duplicates(documents)

    if fuzzy and FUZZY_AVAILABLE:
        to_drop = find_near_duplicates(result, THRESHOLDS.min_duplicate_similarity)
        if to_drop:
            result = [d for d in result if d.metadata.chunk_id not in to_drop]
            logger.info("{} quasi-doublon(s) supprime(s)", len(to_drop))

    if len(result) != initial:
        logger.info(
            "Deduplication : {} -> {} chunks ({:.0%} conserves)",
            initial, len(result), len(result) / initial,
        )
    return result