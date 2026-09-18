"""
Noeuds de l'agent LangGraph.

Chaque noeud est une fonction `(state) -> mise a jour de l'etat`. Ils sont
volontairement MINCES : ils ne font que brancher les briques deja eprouvees de
JangAI (memoire, routage, recherche hybride + rerank, generation). LangGraph
orchestre l'enchainement ; la logique metier reste dans `app.services` /
`app.retrieval`.
"""

from __future__ import annotations

from loguru import logger

from app.agent.state import AgentState
from app.services import conversation
from app.services.generator import generate_answer
from app.services.query_router import detect_filters
from app.services.rag_pipeline import retrieve as pipeline_retrieve


def contextualize_node(state: AgentState) -> dict:
    """Reecrit la question de suivi en question autonome via l'historique.

    Sans historique (1er tour), la question est renvoyee telle quelle. C'est
    cette question autonome qui alimentera le routage et la recherche : les   
    references du type « tout ca » sont ainsi resolues AVANT la recherche.
    """
    standalone = conversation.contextualize(state.get("thread_id"), state["question"])
    return {"standalone_question": standalone}


def _query(state: AgentState) -> str:
    """Question a utiliser pour router et chercher : l'autonome si disponible."""
    return state.get("standalone_question") or state["question"]


def route_node(state: AgentState) -> dict:
    """Detecte les facettes sur la question autonome, avec heritage du contexte.

    Si le suivi ne nomme aucune matiere (« et le chapitre 1 ? »), on herite des
    facettes du tour precedent : la recherche reste sur la meme matiere plutot
    que de partir a la derive.
    """
    detected = detect_filters(_query(state))
    if not detected:
        inherited = conversation.last_filters(state.get("thread_id"))
        if inherited:
            logger.info("Routage : facettes heritees du contexte {}", inherited)
            detected = inherited
    return {"filters": detected}


def retrieve_node(state: AgentState) -> dict:
    """Recherche hybride + reranking, filtree par les facettes detectees."""
    filters = state.get("filters") or None
    candidates = pipeline_retrieve(_query(state), filters=filters)
    logger.info("Agent : {} extraits retenus", len(candidates))
    return {"candidates": candidates}


def generate_node(state: AgentState) -> dict:
    """Genere la reponse citee, en tenant compte de l'historique, puis
    enregistre le tour dans la memoire de la conversation."""
    thread_id = state.get("thread_id")
    history = conversation.history_as_messages(thread_id)
    # On genere sur la question ORIGINALE (ce que l'utilisateur a ecrit),
    # l'historique levant les references ; la recherche, elle, a utilise la
    # question autonome.
    answer = generate_answer(
        state["question"], state.get("candidates") or [], history=history
    )
    conversation.record_turn(
        thread_id, state["question"], answer.text, filters=state.get("filters")
    )
    return {"answer": answer.text, "sources": [s.to_dict() for s in answer.sources]}
