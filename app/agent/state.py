"""
Etat partage de l'agent LangGraph.

C'est la colonne vertebrale du graphe : un dictionnaire type qui circule d'un
noeud a l'autre. Chaque noeud recoit l'etat courant et renvoie une mise a jour
PARTIELLE (LangGraph fusionne). On garde ici les champs de la phase 1 (routage,
recherche, generation) ; les champs agentiques (grade, retries, sous-requetes,
historique) seront ajoutes avec la boucle corrective.

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

    # -- Contextualisation ---------------------------------------------
    # Question de suivi reecrite en question AUTONOME grace a l'historique.
    # C'est elle qui alimente le routage et la recherche.
    standalone_question: str

    # -- Routage (facettes detectees dans la question) -----------------
    # Generique : subject, level, track, et demain document_type, annee...
    filters: dict[str, str]

    # -- Recherche -----------------------------------------------------
    candidates: list[Any]        # list[app.retrieval.search.Candidate]

    # -- Generation ----------------------------------------------------
    answer: str
    sources: list[dict[str, Any]]
