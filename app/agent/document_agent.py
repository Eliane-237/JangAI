"""
Sous-agent specialise : generation d'epreuves en .docx.

Delegue par l'agent principal via l'outil `creer_epreuve`. C'est un petit
graphe LangGraph a lui tout seul :

    plan  ->  generate  ->  render

- `plan`     : recupere les objectifs/competences du PROGRAMME officiel
               (recherche existante) et esquisse la structure de l'epreuve.
- `generate` : redige les exercices ET leur corrige, formules en LaTeX.
- `render`   : convertit en .docx avec equations Word natives (Pandoc).

Les enonces restent ANCRES sur le programme (grounding) ; les mathematiques
sont ecrites en LaTeX pour un rendu propre (exposants, fractions, symboles).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from app.services.document_builder import render_epreuve
from app.services.generator import complete
from app.services.query_router import canonicalize_filters
from app.services.rag_pipeline import retrieve as pipeline_retrieve


class DocState(TypedDict, total=False):
    params: dict[str, Any]   # matiere, niveau, serie, duree, coef, nb_exercices...
    extraits: str            # extraits du programme (grounding)
    plan: str                # plan d'epreuve (texte)
    exercices: list[dict]    # [{points, competence, enonce, corrige}]
    result: dict             # {doc_id, epreuve, corrige} (chemins)
    summary: str             # resume pour l'agent principal


_PLAN_SYSTEM = (
    "Tu es concepteur d'epreuves pour les programmes scolaires senegalais. "
    "A partir des EXTRAITS du programme officiel, tu proposes la structure "
    "d'une epreuve : pour chaque exercice, la competence/chapitre vise (tire "
    "des extraits) et le bareme en points. Reste fidele au programme. Reponds "
    "par une liste simple, sans rediger les enonces."
)

_GEN_SYSTEM = (
    "Tu es concepteur d'epreuves pour les programmes scolaires senegalais. Tu "
    "rediges une epreuve complete ET son corrige, ancres sur les extraits du "
    "programme.\n"
    "REGLES DE FORMAT (imperatives) :\n"
    "- Produis UN bloc par exercice, exactement ainsi :\n"
    "### EXERCICE\n"
    "POINTS: <nombre entier>\n"
    "COMPETENCE: <chapitre ou competence>\n"
    "ENONCE:\n"
    "<enonce en francais>\n"
    "CORRIGE:\n"
    "<corrige detaille>\n"
    "- TOUTES les mathematiques (formules, exposants, fractions, symboles) en "
    "LaTeX entre $...$ (ou $$...$$ pour une formule isolee). Exemple : "
    "$f(x)=x^{3}-3x+1$, $u_{n+1}=\\frac{1}{2}u_n+3$, $P(A\\cup B)$.\n"
    "- N'ecris RIEN en dehors des blocs."
)


def _query_from(params: dict) -> tuple[str, dict]:
    """Construit la requete et les filtres de recherche a partir des parametres."""
    matiere = params.get("matiere", "")
    niveau = params.get("niveau", "")
    query = f"programme {matiere} {niveau} objectifs competences contenus".strip()
    filters = canonicalize_filters(
        {
            "subject": matiere or None,
            "level": niveau or None,
            "track": params.get("serie") or None,
        }
    )
    return query, filters


def _retrieve_programme(query: str, filters: dict):
    """Recherche le programme en RELACHANT les filtres si rien ne remonte.

    La serie (puis le niveau) sont trop granulaires pour *trouver* le programme
    et peuvent tout exclure (ex. serie « s » alors que les chunks sont « ls »).
    On tente du plus precis au plus large plutot que de rendre 0 extrait.
    """
    attempts = [
        filters,
        {k: v for k, v in filters.items() if k != "track"},
        {"subject": filters["subject"]} if filters.get("subject") else {},
        {},
    ]
    tried: list[dict] = []
    for attempt in attempts:
        clean = {k: v for k, v in attempt.items() if v}
        if clean in tried:
            continue
        tried.append(clean)
        candidates = pipeline_retrieve(query, filters=clean or None)
        if candidates:
            return candidates, clean
    return [], {}


def plan_node(state: DocState) -> dict:
    """Recupere le programme et esquisse la structure de l'epreuve."""
    params = state["params"]
    query, filters = _query_from(params)
    candidates, used = _retrieve_programme(query, filters)
    extraits = "\n\n".join(c.content for c in candidates[:8])[:6000]
    logger.info(
        "Sous-agent epreuve : {} extraits de programme (filtres {})",
        len(candidates), used,
    )

    user = (
        f"Matiere : {params.get('matiere')}\nNiveau : {params.get('niveau')} "
        f"{params.get('serie', '')}\nNombre d'exercices : {params.get('nb_exercices', 3)}\n"
        f"Bareme total : {params.get('total_points', 20)} points\n"
        f"Difficulte : {params.get('difficulte', 'standard')}\n\n"
        f"EXTRAITS DU PROGRAMME :\n{extraits or '(aucun extrait trouve)'}\n\n"
        "Propose le plan de l'epreuve (un exercice par ligne : competence + points)."
    )
    plan = complete(_PLAN_SYSTEM, user, temperature=0.3, max_tokens=1200)
    return {"extraits": extraits, "plan": plan}


def _parse_exercices(text: str) -> list[dict]:
    """Analyse la sortie delimitee du LLM en liste d'exercices structures."""
    exercices: list[dict] = []
    for block in re.split(r"#*\s*EXERCICE\s*\n", text)[1:]:
        m_pts = re.search(r"POINTS\s*:\s*(\d+)", block, re.I)
        m_comp = re.search(r"COMPETENCE\s*:\s*(.+)", block, re.I)
        m_en = re.search(r"ENONCE\s*:\s*(.*?)\n\s*CORRIGE\s*:", block, re.I | re.S)
        m_cor = re.search(r"CORRIGE\s*:\s*(.*)", block, re.I | re.S)
        enonce = (m_en.group(1).strip() if m_en else "")
        if not enonce:
            continue
        exercices.append(
            {
                "points": int(m_pts.group(1)) if m_pts else 0,
                "competence": (m_comp.group(1).strip() if m_comp else ""),
                "enonce": enonce,
                "corrige": (m_cor.group(1).strip() if m_cor else ""),
            }
        )
    return exercices


def generate_node(state: DocState) -> dict:
    """Redige les exercices et leur corrige (formules en LaTeX)."""
    params = state["params"]
    user = (
        f"Matiere : {params.get('matiere')}\nNiveau : {params.get('niveau')} "
        f"{params.get('serie', '')}\nBareme total vise : "
        f"{params.get('total_points', 20)} points\n\n"
        f"PLAN PROPOSE :\n{state.get('plan', '')}\n\n"
        f"EXTRAITS DU PROGRAMME :\n{state.get('extraits', '')}\n\n"
        "Redige maintenant l'epreuve complete au format impose (blocs EXERCICE)."
    )
    raw = complete(_GEN_SYSTEM, user, temperature=0.4, max_tokens=6000)
    exercices = _parse_exercices(raw)
    logger.info("Sous-agent epreuve : {} exercices rediges", len(exercices))
    return {"exercices": exercices}


def render_node(state: DocState) -> dict:
    """Rend l'epreuve et le corrige en .docx."""
    params = state["params"]
    exercices = state.get("exercices") or []
    data = {
        "etablissement": params.get("etablissement", ""),
        "matiere": params.get("matiere", ""),
        "niveau": params.get("niveau", ""),
        "serie": params.get("serie", ""),
        "duree": params.get("duree", ""),
        "coef": params.get("coef", ""),
        "consignes": params.get("consignes", ""),
        "exercices": exercices,
        "total_points": sum(ex["points"] for ex in exercices),
    }
    result = render_epreuve(data)
    summary = (
        f"Epreuve de {data['matiere']} {data['niveau']} generee : "
        f"{len(exercices)} exercices, {data['total_points']} points."
    )
    return {
        "result": {
            "doc_id": result["doc_id"],
            "epreuve": str(result["epreuve"]),
            "corrige": str(result["corrige"]),
        },
        "summary": summary,
    }


@lru_cache(maxsize=1)
def _graph():
    builder = StateGraph(DocState)
    builder.add_node("plan", plan_node)
    builder.add_node("generate", generate_node)
    builder.add_node("render", render_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "generate")
    builder.add_edge("generate", "render")
    builder.add_edge("render", END)
    return builder.compile()


def run(params: dict) -> dict:
    """Genere une epreuve + corrige a partir des parametres.

    Returns:
        {summary, doc_id, epreuve, corrige} : resume et chemins des .docx.
    """
    final = _graph().invoke({"params": params})
    result = final.get("result", {})
    return {"summary": final.get("summary", ""), **result}
