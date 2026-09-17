"""
Graphe de l'agent RAG (LangGraph).

Phase 1 — flux LINEAIRE, pour valider l'ossature :

    START -> route -> retrieve -> generate -> END

`route` detecte les facettes (matiere, niveau...), `retrieve` cherche +
rerank en les filtrant, `generate` produit la reponse citee. La boucle
corrective (grade -> reformuler -> re-chercher) sera ajoutee en phase 2 :
il suffira d'inserer des noeuds et des aretes conditionnelles, l'etat et les
noeuds actuels restant inchanges.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import generate_node, retrieve_node, route_node
from app.agent.state import AgentState


def build_graph():
    """Construit et compile le graphe de l'agent."""
    builder = StateGraph(AgentState)
    builder.add_node("route", route_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "route")
    builder.add_edge("route", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_agent():
    """Instance unique du graphe compile (charge une fois par processus)."""
    return build_graph()


def answer(question: str) -> dict:
    """Execute l'agent sur une question et renvoie reponse + sources.

    Args:
        question: Question de l'utilisateur

    Returns:
        Dictionnaire {answer, sources, filters}
    """
    final_state = get_agent().invoke({"question": question})
    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "filters": final_state.get("filters", {}),
    }
