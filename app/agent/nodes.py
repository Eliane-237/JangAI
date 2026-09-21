"""
Noeuds de l'agent LangGraph (architecture a outils).

Le LLM est un AGENT : il decide lui-meme d'appeler l'outil de recherche ou de
repondre directement. Deux noeuds suffisent, relies par une boucle :

    agent  --(le LLM demande l'outil ?)-->  tools  --> agent
      |                                                   |
      +--(non : reponse prete)--> END <-------------------+

`agent` fait parler le LLM (avec l'outil disponible) ; `tools` execute les
recherches demandees et renvoie les extraits ; on reboucle jusqu'a ce que le
LLM produise sa reponse finale. Le bavardage (« bonjour », « merci ») ne
declenche aucun appel d'outil : l'agent repond en un seul tour.
"""

from __future__ import annotations

from langgraph.graph import END
from loguru import logger

from app.agent.state import AgentState
from app.agent.tools import TOOLS, execute_search
from app.prompts.templates import AGENT_SYSTEM_PROMPT
from app.services import conversation
from app.services.generator import chat_with_tools, complete

# Garde-fou : nombre maximum de rondes d'outils par tour. Au-dela, on force le
# LLM a conclure (tool_choice="none") pour eviter toute boucle infinie.
MAX_TOOL_ROUNDS = 4


def _initial_messages(state: AgentState) -> list[dict]:
    """Amorce le dialogue interne : persona + historique + question du tour."""
    messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    for turn in conversation.history_as_messages(state.get("thread_id")):
        if turn.get("question"):
            messages.append({"role": "user", "content": turn["question"]})
        if turn.get("answer"):
            messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": state["question"]})
    return messages


def _assistant_to_dict(message) -> dict:
    """Serialise le message de l'assistant (avec ses eventuels appels d'outils)."""
    payload: dict = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ]
    return payload


def _finalize(state: AgentState, messages: list[dict], answer: str) -> dict:
    """Cloture le tour : memorise la conversation et renvoie la reponse."""
    answer = (answer or "").strip()
    conversation.record_turn(
        state.get("thread_id"), state["question"], answer, filters=state.get("filters")
    )
    logger.info("Agent : reponse finale ({} caracteres)", len(answer))
    return {"messages": messages, "pending": [], "answer": answer}


def _conclude_without_tools(state: AgentState, messages: list[dict]) -> str:
    """Force une reponse en prose, SANS outil.

    On repart d'une conversation PROPRE (sans historique d'appels d'outils, qui
    incite gpt-oss a re-chercher indefiniment) et on inline les extraits deja
    recuperes. Sans amorce d'outil ni parametre `tools`, le modele conclut.
    """
    gathered = "\n\n".join(
        m["content"] for m in messages if m.get("role") == "tool" and m.get("content")
    )
    user = state["question"]
    if gathered:
        user += (
            "\n\n(Reponds maintenant naturellement, en prose, a partir des "
            "extraits ci-dessous, sans chercher davantage et sans afficher de "
            "numeros de source. Signale honnetement ce qui manque.)"
            f"\n\n{gathered}"
        )
    else:
        user += (
            "\n\n(Aucun extrait pertinent n'a ete trouve. Reponds honnetement que "
            "tu n'as pas l'information dans les programmes, sans inventer.)"
        )
    return complete(AGENT_SYSTEM_PROMPT, user)


def agent_node(state: AgentState) -> dict:
    """Fait parler le LLM ; il repond, ou demande une (des) recherche(s)."""
    messages = state.get("messages") or _initial_messages(state)

    # Plafond de rondes atteint : on conclut sans outil (garde-fou anti-boucle,
    # ex. une recherche qui reste vide et que le modele s'obstine a relancer).
    rounds = sum(1 for m in messages if m.get("role") == "tool")
    if rounds >= MAX_TOOL_ROUNDS:
        logger.info("Agent : plafond d'outils atteint, conclusion forcee.")
        return _finalize(state, messages, _conclude_without_tools(state, messages))

    try:
        message = chat_with_tools(messages, TOOLS, tool_choice="auto")
    except Exception as exc:
        # Ex. gpt-oss emet un appel d'outil mal forme (400 tool_use_failed) :
        # on conclut sans outil plutot que de faire echouer le tour.
        logger.warning("Appel d'outil en echec ({}) : conclusion sans outil.", exc)
        return _finalize(state, messages, _conclude_without_tools(state, messages))

    messages = messages + [_assistant_to_dict(message)]

    if message.tool_calls:
        pending = [
            {"id": call.id, "arguments": call.function.arguments}
            for call in message.tool_calls
        ]
        logger.info("Agent : {} recherche(s) demandee(s)", len(pending))
        return {"messages": messages, "pending": pending}

    return _finalize(state, messages, message.content)


def tools_node(state: AgentState) -> dict:
    """Execute les recherches demandees et renvoie les extraits au LLM."""
    messages = list(state["messages"])
    sources = list(state.get("sources") or [])
    filters = dict(state.get("filters") or {})

    for call in state.get("pending") or []:
        content, new_sources, used_filters = execute_search(
            call["arguments"], start_index=len(sources) + 1
        )
        sources.extend(source.to_dict() for source in new_sources)
        if used_filters:
            filters = used_filters
        messages.append(
            {"role": "tool", "tool_call_id": call["id"], "content": content}
        )

    return {
        "messages": messages,
        "pending": [],
        "sources": sources,
        "filters": filters,
    }


def should_continue(state: AgentState) -> str:
    """Aiguillage : vers les outils si le LLM en a demande, sinon fin."""
    return "tools" if state.get("pending") else END
