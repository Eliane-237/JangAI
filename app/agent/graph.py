"""
Graphe de l'agent RAG (LangGraph).

Flux LINEAIRE, desormais conversationnel :

    START -> contextualize -> route -> retrieve -> generate -> END

`contextualize` reecrit la question de suivi en question autonome grace a la
memoire de la conversation (thread_id), `route` detecte les facettes, `retrieve`
cherche + rerank en les filtrant, `generate` produit la reponse citee et
enregistre le tour. La boucle corrective (grade -> reformuler -> re-chercher)
s'inserera en phase 2 : l'etat et les noeuds actuels restent inchanges.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    contextualize_node,
    generate_node,
    retrieve_node,
    route_node,
)
from app.agent.state import AgentState
from app.services import conversation


def build_graph():
    """Construit et compile le graphe de l'agent."""
    builder = StateGraph(AgentState)
    builder.add_node("contextualize", contextualize_node)
    builder.add_node("route", route_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "route")
    builder.add_edge("route", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_agent():
    """Instance unique du graphe compile (charge une fois par processus)."""
    return build_graph()


def answer(question: str, thread_id: str | None = None) -> dict:
    """Execute l'agent sur une question et renvoie reponse + sources.

    Args:
        question: Question de l'utilisateur
        thread_id: Identifiant de conversation. Absent -> une nouvelle
            conversation est ouverte et son identifiant est renvoye, pour que
            le client l'envoie aux tours suivants.

    Returns:
        Dictionnaire {answer, sources, filters, thread_id}
    """
    thread_id = thread_id or conversation.new_thread_id()
    final_state = get_agent().invoke({"question": question, "thread_id": thread_id})
    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "filters": final_state.get("filters", {}),
        "thread_id": thread_id,
    }
