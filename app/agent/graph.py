"""
Graphe de l'agent RAG (LangGraph), architecture AGENTIQUE a outils.

    START -> agent  --(le LLM demande l'outil ?)-->  tools  --> agent
                |                                                  |
                +--(non : reponse prete)--> END <-----------------+

Le noeud `agent` (le LLM) decide lui-meme : bavarder, ou appeler l'outil
`chercher_programme`. C'est cette decision qui remplace l'ancien aiguillage
code a la main (contextualize -> route -> retrieve). L'agent peut chercher
PLUSIEURS fois avant de conclure (questions complexes / multi-matieres), puis
`generate` disparait : la reponse finale est produite par l'agent lui-meme.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import agent_node, should_continue, tools_node
from app.agent.state import AgentState
from app.services import conversation


def build_graph():
    """Construit et compile le graphe de l'agent."""
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "agent")
    # Apres l'agent : soit executer les outils demandes, soit terminer.
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    # Apres les outils : retour a l'agent pour exploiter les extraits.
    builder.add_edge("tools", "agent")

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
