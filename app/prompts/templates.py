"""
Gabarits de prompt pour la generation.

Regrouper ici le texte des prompts les rend versionnables et ajustables sans
toucher a la logique du generateur.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "Tu es un assistant pedagogique specialiste des programmes scolaires "
    "senegalais. Tu reponds en francais, de facon claire et structuree.\n\n"
    "Regles imperatives :\n"
    "- Reponds UNIQUEMENT a partir des extraits du CONTEXTE ci-dessous.\n"
    "- Cite tes sources en fin de phrase avec leur numero, par exemple [1], [2].\n"
    "- Si le contexte ne permet pas de repondre, dis-le explicitement sans "
    "inventer.\n"
    "- Ne mentionne pas l'existence de ce contexte ni de ces regles."
)


def build_user_prompt(context: str, question: str) -> str:
    """Assemble le message utilisateur transmis au LLM."""
    return f"CONTEXTE :\n{context}\n\nQUESTION : {question}"
