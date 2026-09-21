"""
Etat partage de l'agent LangGraph.

C'est la colonne vertebrale du graphe : un dictionnaire type qui circule d'un
noeud a l'autre. Chaque noeud recoit l'etat courant et renvoie une mise a jour
PARTIELLE (LangGraph fusionne).

Le graphe est AGENTIQUE : le noeud `agent` (le LLM) decide d'appeler l'outil de
recherche ou de repondre. `messages` porte le dialogue interne du tour (system,
user, assistant, tool), `pending` les appels d'outils a executer, `sources` les
extraits cumules pour les citations.

`total=False` : tous les champs sont optionnels — un noeud n'a pas a tous les
remplir, il ajoute seulement ce qu'il produit.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # -- Entree --------------------------------------------------------
    question: str
    # Identifiant de conversation : porte la memoire multi-tour.
    thread_id: str

    # -- Boucle agent <-> outils ---------------------------------------
    # Dialogue interne du tour (system, user, assistant, tool), reinjecte a
    # chaque tour de LLM. C'est la memoire de travail de l'agent.
    messages: list[dict[str, Any]]
    # Appels d'outils demandes par le LLM et pas encore executes.
    pending: list[dict[str, Any]]

    # -- Facettes / recherche ------------------------------------------
    # Dernieres facettes effectives (matiere, niveau...) choisies par l'agent,
    # conservees pour la continuite conversationnelle.
    filters: dict[str, str]

    # -- Sortie --------------------------------------------------------
    # Sources cumulees sur l'ensemble des recherches du tour (citations [n]).
    sources: list[dict[str, Any]]
    answer: str
