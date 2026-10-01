"""
Rendu de documents Word (.docx), avec formules mathematiques natives.

Etape DETERMINISTE de la generation : le contenu (epreuve, corrige...) est
produit en amont sous forme de donnees structurees, ou les formules sont
ecrites en LaTeX (`$...$`). On assemble un Markdown puis on le convertit en
.docx via Pandoc, qui transforme le LaTeX en EQUATIONS Word natives (OMML,
editables) plutot qu'en texte brut ou en images.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pypandoc
from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = _ROOT / "generated"
TEMPLATES_DIR = _ROOT / "templates"
# Document de reference Pandoc : porte l'en-tete officiel et les styles.
REFERENCE_DOCX = TEMPLATES_DIR / "reference.docx"


def _header_markdown(data: dict, titre: str) -> list[str]:
    """En-tete commun (etablissement, titre, classe/duree/coef, consignes)."""
    lines: list[str] = []
    if data.get("etablissement"):
        lines.append(f"**{data['etablissement']}**")
        lines.append("")
    lines.append(f"# {titre} de {data.get('matiere', '')}".rstrip())
    lines.append("")
    infos = [
        f"**Classe :** {data.get('niveau', '')} {data.get('serie', '')}".rstrip(),
        f"**Durée :** {data.get('duree', '')}",
        f"**Coefficient :** {data.get('coef', '')}",
    ]
    lines.append("  —  ".join(i for i in infos if i.split(":**")[-1].strip()))
    lines.append("")
    if data.get("consignes"):
        lines.append(f"*Consignes : {data['consignes']}*")
        lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def _bareme_markdown(exercices: list[dict], total: int) -> list[str]:
    """Tableau de bareme (Markdown) : un exercice par ligne + total."""
    lines = ["## Barème", "", "| Exercice | Points |", "|:---|:---:|"]
    for i, ex in enumerate(exercices, start=1):
        competence = ex.get("competence", "")
        libelle = f"Exercice {i}" + (f" — {competence}" if competence else "")
        lines.append(f"| {libelle} | {ex.get('points', 0)} |")
    lines.append(f"| **Total** | **{total}** |")
    lines.append("")
    return lines


def _epreuve_markdown(data: dict) -> str:
    """Assemble le Markdown de l'epreuve (enonces, formules LaTeX conservees)."""
    lines = _header_markdown(data, "Épreuve")
    exercices = data.get("exercices", [])
    for i, ex in enumerate(exercices, start=1):
        competence = ex.get("competence", "")
        titre = f"## Exercice {i} ({ex.get('points', 0)} points)"
        if competence:
            titre += f" — {competence}"
        lines.append(titre)
        lines.append("")
        lines.append(ex.get("enonce", "").strip())
        lines.append("")
    total = data.get("total_points") or sum(ex.get("points", 0) for ex in exercices)
    lines.append("---")
    lines.append("")
    lines.extend(_bareme_markdown(exercices, total))
    return "\n".join(lines)


def _corrige_markdown(data: dict) -> str:
    """Assemble le Markdown du corrige (solutions, formules LaTeX conservees)."""
    lines = _header_markdown(data, "Corrigé")
    for i, ex in enumerate(data.get("exercices", []), start=1):
        competence = ex.get("competence", "")
        titre = f"## Exercice {i} ({ex.get('points', 0)} points)"
        if competence:
            titre += f" — {competence}"
        lines.append(titre)
        lines.append("")
        lines.append(ex.get("corrige", "").strip() or "_Corrigé non disponible._")
        lines.append("")
    return "\n".join(lines)


def _render_markdown(markdown: str, name: str) -> Path:
    """Convertit un Markdown (avec LaTeX) en .docx via Pandoc."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    if not name.endswith(".docx"):
        name += ".docx"
    out_path = GENERATED_DIR / name
    # Document de reference : en-tete officiel + styles (s'il a ete genere).
    extra_args = []
    if REFERENCE_DOCX.exists():
        extra_args = ["--reference-doc", str(REFERENCE_DOCX)]
    # Pandoc transforme `$...$` en equations Word natives (OMML).
    pypandoc.convert_text(
        markdown, "docx", format="markdown", outputfile=str(out_path),
        extra_args=extra_args,
    )
    logger.info("Document genere : {}", out_path)
    return out_path


def render_epreuve(data: dict, doc_id: str | None = None) -> dict:
    """Rend l'epreuve ET son corrige en .docx.

    Args:
        data: etablissement, matiere, niveau, serie, duree, coef, consignes,
            et `exercices` (liste {points, competence, enonce, corrige}) ou
            enonce/corrige peuvent contenir du LaTeX `$...$`.
        doc_id: Identifiant commun aux deux fichiers (sinon genere).

    Returns:
        {doc_id, epreuve, corrige} : l'identifiant et les deux chemins.
    """
    doc_id = doc_id or uuid.uuid4().hex[:8]
    epreuve = _render_markdown(_epreuve_markdown(data), f"epreuve_{doc_id}.docx")
    corrige = _render_markdown(_corrige_markdown(data), f"corrige_{doc_id}.docx")
    return {"doc_id": doc_id, "epreuve": epreuve, "corrige": corrige}
