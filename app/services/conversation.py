"""
Memoire conversationnelle multi-tour.

Deux responsabilites :

    1. MEMOIRE. Conserver, par conversation (`thread_id`), les derniers tours
       (question de l'utilisateur + reponse de l'assistant). Stockage en memoire
       du processus, borne en taille : suffisant pour un serveur unique. Pour
       du multi-processus ou de la persistance, on remplacerait `_STORE` par
       Redis/PostgreSQL sans toucher au reste.

    2. CONTEXTUALISATION. Reecrire une question de suivi en une question
       AUTONOME a l'aide de l'historique. C'est ce qui corrige la derive : sans
       elle, « tout ca c'est le chapitre un ? » est cherche seul et part sur les
       maths ; reecrite en « Les objectifs de lecture en francais constituent-ils
       le chapitre 1 ? », le routage et la recherche restent sur le francais.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass

from loguru import logger

from app.config import get_settings


@dataclass
class Turn:
    """Un tour de conversation."""

    question: str
    answer: str


# Store {thread_id -> file d'attente bornee de Turn}, protege par un verrou.
# OrderedDict pour evincer les conversations les plus anciennes (LRU).
_LOCK = threading.Lock()
_STORE: "OrderedDict[str, deque[Turn]]" = OrderedDict()
# Dernieres facettes effectives par conversation (matiere, niveau...), pour la
# continuite : un suivi qui ne nomme pas de matiere herite de la precedente.
_FILTERS: dict[str, dict[str, str]] = {}


# ======================================================================
# Memoire
# ======================================================================


def new_thread_id() -> str:
    """Genere un identifiant de conversation neuf."""
    return uuid.uuid4().hex[:16]


def get_history(thread_id: str | None) -> list[Turn]:
    """Tours precedents d'une conversation (vides si inconnue)."""
    if not thread_id:
        return []
    with _LOCK:
        turns = _STORE.get(thread_id)
        return list(turns) if turns else []


def record_turn(
    thread_id: str | None,
    question: str,
    answer: str,
    filters: dict[str, str] | None = None,
) -> None:
    """Ajoute un tour a la conversation et evince les plus anciennes.

    `filters` (facettes effectives du tour) sont memorisees pour la continuite :
    le tour suivant en heritera s'il ne nomme aucune matiere.
    """
    if not thread_id:
        return
    settings = get_settings()
    with _LOCK:
        turns = _STORE.get(thread_id)
        if turns is None:
            turns = deque(maxlen=settings.conversation_max_turns)
            _STORE[thread_id] = turns
        turns.append(Turn(question=question, answer=answer))
        _STORE.move_to_end(thread_id)
        if filters:
            _FILTERS[thread_id] = dict(filters)
        while len(_STORE) > settings.conversation_max_threads:
            old, _ = _STORE.popitem(last=False)
            _FILTERS.pop(old, None)


def last_filters(thread_id: str | None) -> dict[str, str]:
    """Dernieres facettes effectives de la conversation (vide si inconnue)."""
    if not thread_id:
        return {}
    with _LOCK:
        return dict(_FILTERS.get(thread_id, {}))


def history_as_messages(thread_id: str | None) -> list[dict]:
    """Historique au format attendu par le generateur ({question, answer})."""
    return [{"question": t.question, "answer": t.answer} for t in get_history(thread_id)]


# ======================================================================
# Contextualisation (reecriture de la question de suivi)
# ======================================================================

_REWRITE_SYSTEM = (
    "Tu reecris la DERNIERE question d'un utilisateur pour qu'elle soit complete "
    "et comprehensible SANS l'historique de conversation. Resous les references "
    "implicites (\"ca\", \"tout ca\", \"et pour ...\", \"le premier\"...) en "
    "t'appuyant sur l'historique.\n"
    "IMPORTANT : conserve EXPLICITEMENT la matiere et le niveau evoques (par "
    "exemple, si l'echange porte sur le francais en terminale, la question "
    "reecrite doit contenir les mots 'francais' et 'terminale'). Si l'utilisateur "
    "change de matiere, utilise la NOUVELLE matiere.\n"
    "Conserve la meme langue et l'intention exacte. Si la question est deja "
    "autonome, renvoie-la a l'identique. Reponds UNIQUEMENT par la question "
    "reecrite, sans guillemets ni commentaire."
)


def contextualize(thread_id: str | None, question: str) -> str:
    """Reecrit `question` en une question autonome a partir de l'historique.

    Sans historique, la question est renvoyee telle quelle (aucun appel LLM).
    En cas d'echec du LLM, on retombe sur la question brute : la contextualisation
    ne doit jamais bloquer une reponse.

    Args:
        thread_id: Identifiant de conversation
        question: Question de suivi de l'utilisateur

    Returns:
        Question autonome, prete pour le routage et la recherche
    """
    history = get_history(thread_id)
    if not history:
        return question

    transcript = "\n".join(
        f"Utilisateur: {t.question}\nAssistant: {t.answer}" for t in history
    )
    user = (
        f"Historique de conversation :\n{transcript}\n\n"
        f"Derniere question : {question}\n\nQuestion reecrite :"
    )

    # Import differe : evite un cycle (generator importe la recherche, etc.).
    from app.services.generator import complete

    try:
        # Budget genereux : les modeles a raisonnement (gpt-oss) consomment des
        # tokens avant de repondre ; trop peu tronquerait la reecriture.
        rewritten = complete(_REWRITE_SYSTEM, user, temperature=0.0, max_tokens=512)
        rewritten = rewritten.strip().strip('"').strip()
        # Validation : une reecriture credible fait au moins quelques mots. Une
        # sortie tronquee (« Est-ce que ») est rejetee au profit de la question
        # brute — la continuite reste assuree par les filtres herites.
        if len(rewritten) >= 12 and len(rewritten.split()) >= 3:
            if rewritten.lower() != question.lower():
                logger.info("Contextualisation : '{}' -> '{}'", question, rewritten)
            return rewritten
        logger.warning("Reecriture rejetee ('{}') : question brute.", rewritten)
    except Exception as exc:  # pragma: no cover
        logger.warning("Contextualisation impossible ({}) : question brute.", exc)
    return question
