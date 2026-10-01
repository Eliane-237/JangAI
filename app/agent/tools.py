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

# Noms des outils, partages entre les schemas et l'aiguillage d'execution.
SEARCH_TOOL_NAME = "chercher_programme"
EPREUVE_TOOL_NAME = "creer_epreuve"

# Schemas au format OpenAI/Groq. La description guide le LLM : quand appeler
# chaque outil, et comment remplir les champs.
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
    },
    {
        "type": "function",
        "function": {
            "name": EPREUVE_TOOL_NAME,
            "description": (
                "Genere une EPREUVE (sujet + corrige) au format Word .docx, "
                "ancree sur le programme officiel, avec les formules en notation "
                "mathematique. A utiliser quand l'enseignant demande de creer / "
                "generer / preparer une epreuve, un devoir ou un sujet d'examen. "
                "Renvoie des liens de telechargement a transmettre a l'enseignant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "matiere": {"type": "string", "description": "Matiere (ex. mathematiques)."},
                    "niveau": {"type": "string", "description": "Niveau (ex. terminale)."},
                    "serie": {"type": "string", "description": "Serie/filiere si precisee (ex. S, L)."},
                    "duree": {"type": "string", "description": "Duree (ex. « 4 heures »)."},
                    "coef": {"type": "string", "description": "Coefficient si precise."},
                    "nb_exercices": {"type": "integer", "description": "Nombre d'exercices (defaut 3)."},
                    "total_points": {"type": "integer", "description": "Bareme total (defaut 20)."},
                    "difficulte": {"type": "string", "description": "Niveau de difficulte vise."},
                    "consignes": {"type": "string", "description": "Consignes a afficher en en-tete."},
                    "etablissement": {"type": "string", "description": "Nom de l'etablissement si fourni."},
                },
                "required": ["matiere", "niveau"],
            },
        },
    },
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


def execute_creer_epreuve(arguments: str | dict) -> str:
    """Genere une epreuve + corrige et renvoie un message avec les liens.

    Delegue au sous-agent `document_agent` (plan -> redaction -> rendu .docx),
    puis renvoie au LLM les liens de telechargement a transmettre a l'enseignant.
    """
    # Import differe : evite de charger le sous-agent/pandoc quand on ne
    # genere pas de document.
    from app.agent import document_agent

    args = _parse_arguments(arguments)
    if not (args.get("matiere") and args.get("niveau")):
        return "Impossible de generer l'epreuve : precise au moins la matiere et le niveau."

    params = {
        "matiere": args.get("matiere", ""),
        "niveau": args.get("niveau", ""),
        "serie": args.get("serie", ""),
        "duree": args.get("duree", ""),
        "coef": args.get("coef", ""),
        "nb_exercices": args.get("nb_exercices", 3),
        "total_points": args.get("total_points", 20),
        "difficulte": args.get("difficulte", "standard"),
        "consignes": args.get("consignes", ""),
        "etablissement": args.get("etablissement", ""),
    }
    result = document_agent.run(params)
    epreuve = _basename(result.get("epreuve", ""))
    corrige = _basename(result.get("corrige", ""))
    logger.info("Outil {} : {}", EPREUVE_TOOL_NAME, result.get("summary"))
    return (
        f"{result.get('summary', 'Epreuve generee.')}\n"
        f"Transmets ces liens de telechargement a l'enseignant :\n"
        f"- Sujet : /documents/{epreuve}\n"
        f"- Corrige : /documents/{corrige}"
    )


def _basename(path: str) -> str:
    """Nom de fichier seul (pour construire l'URL /documents/<nom>)."""
    from pathlib import Path

    return Path(path).name if path else ""


def execute_tool(name: str, arguments: str | dict, start_index: int) -> tuple[str, list[Source], dict]:
    """Aiguille un appel d'outil vers son executeur.

    Returns:
        (message `tool` pour le LLM, sources produites, facettes utilisees).
        Les outils sans sources (generation de document) renvoient [] et {}.
    """
    if name == SEARCH_TOOL_NAME:
        return execute_search(arguments, start_index)
    if name == EPREUVE_TOOL_NAME:
        return execute_creer_epreuve(arguments), [], {}
    logger.warning("Outil inconnu demande par le LLM : {}", name)
    return f"Outil inconnu : {name}.", [], {}
