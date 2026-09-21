"""
Outils confies a l'agent conversationnel.

L'agent (le LLM) ne cherche pas « en dur » : on lui expose un OUTIL,
`chercher_programme`, et c'est lui qui decide de l'appeler. L'outil n'est
qu'un mince adaptateur autour de la pipeline eprouvee de JangAI (recherche
hybride + reranking) : toute la qualite de recuperation reste dans
`app.retrieval`, on l'emballe seulement pour le LLM.
"""

from __future__ import annotations

import json

from loguru import logger

from app.config import get_settings
from app.prompts.templates import format_tool_results
from app.services.generator import Source, build_context
from app.services.query_router import canonicalize_filters
from app.services.rag_pipeline import retrieve as pipeline_retrieve

# Nom de l'outil, partage entre le schema et l'aiguillage d'execution.
SEARCH_TOOL_NAME = "chercher_programme"

# Schema au format OpenAI/Groq. La description guide le LLM : quand appeler
# l'outil, et comment remplir les facettes (matiere / niveau / serie).
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": SEARCH_TOOL_NAME,
            "description": (
                "Recherche dans la base officielle des programmes scolaires "
                "senegalais et renvoie des extraits numerotes et cites. A "
                "utiliser pour toute question sur les objectifs, contenus, "
                "competences, chapitres ou horaires d'une matiere. Pour une "
                "question a plusieurs volets, appeler l'outil une fois par volet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "requete": {
                        "type": "string",
                        "description": (
                            "Requete de recherche claire et AUTONOME (references "
                            "du fil deja resolues), ex. « objectifs de lecture en "
                            "francais en terminale »."
                        ),
                    },
                    "matiere": {
                        "type": "string",
                        "description": "Filtre matiere si connu (ex. francais, mathematiques).",
                    },
                    "niveau": {
                        "type": "string",
                        "description": "Filtre niveau si connu (ex. terminale, seconde).",
                    },
                    "serie": {
                        "type": "string",
                        "description": "Filtre serie/filiere si connu (ex. S, L).",
                    },
                },
                "required": ["requete"],
            },
        },
    }
]


def _parse_arguments(raw: str | dict) -> dict:
    """Decode les arguments d'un appel d'outil (JSON renvoye par le LLM)."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        logger.warning("Arguments d'outil illisibles : {}", raw)
        return {}


def execute_search(arguments: str | dict, start_index: int) -> tuple[str, list[Source], dict]:
    """Execute une recherche demandee par l'agent.

    Args:
        arguments: Arguments JSON produits par le LLM (`requete`, `matiere`...)
        start_index: Numero du premier extrait, pour une numerotation CONTINUE
            entre plusieurs recherches d'un meme tour.

    Returns:
        Tuple (message `tool` pour le LLM, sources produites, facettes utilisees)
    """
    args = _parse_arguments(arguments)
    query = (args.get("requete") or "").strip()
    if not query:
        return format_tool_results(""), [], {}

    requested = {
        key: value.strip()
        for key, value in (
            ("subject", args.get("matiere")),
            ("level", args.get("niveau")),
            ("track", args.get("serie")),
        )
        if isinstance(value, str) and value.strip()
    }
    # Le LLM ecrit « mathematiques » / « terminale L » ; la base stocke
    # « maths » / « terminale ». On aligne sur les vraies valeurs, sinon on
    # laisse tomber le filtre : mieux vaut chercher large que filtrer a zero.
    filters = canonicalize_filters(requested)

    candidates = pipeline_retrieve(query, filters=filters or None)
    logger.info(
        "Outil {} : '{}' (facettes={}) -> {} extraits",
        SEARCH_TOOL_NAME, query, filters, len(candidates),
    )
    context, sources = build_context(
        candidates, get_settings().max_context_chars, start_index=start_index
    )
    return format_tool_results(context), sources, filters
