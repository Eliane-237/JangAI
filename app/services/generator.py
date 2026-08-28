"""
Generation de la reponse par un LLM (Groq).

Le modele ne repond QUE sur la base des extraits fournis, et cite ses sources.
Cette contrainte est ce qui distingue une reponse tracable d'une hallucination :
chaque affirmation renvoie a un extrait numerote, rattache a un fichier et une
page.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from app.config import get_settings
from app.prompts.templates import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.search import Candidate


@dataclass
class Source:
    """Une source citee dans la reponse."""

    index: int
    source_file: str
    page_number: int | None
    hierarchy_path: list[str]
    subject: str | None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "source_file": self.source_file,
            "page": self.page_number,
            "section": " > ".join(self.hierarchy_path),
            "subject": self.subject,
        }


@dataclass
class Answer:
    """Reponse generee et ses sources."""

    text: str
    sources: list[Source]

    def to_dict(self) -> dict:
        return {"answer": self.text, "sources": [s.to_dict() for s in self.sources]}


def build_context(candidates: list[Candidate], max_chars: int) -> tuple[str, list[Source]]:
    """Assemble le contexte numerote et la liste des sources.

    Args:
        candidates: Candidats retenus, deja ordonnes
        max_chars: Budget de caracteres pour le contexte

    Returns:
        Tuple (contexte textuel, sources)
    """
    blocks: list[str] = []
    sources: list[Source] = []
    used = 0

    for index, candidate in enumerate(candidates, start=1):
        header = candidate.hierarchy_text()
        location = f"{candidate.source_file}, p.{candidate.page_number or '?'}"
        prefix = f"[{index}] ({location}"
        prefix += f" — {header})" if header else ")"
        block = f"{prefix}\n{candidate.content}"

        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
        sources.append(
            Source(
                index=index,
                source_file=candidate.source_file,
                page_number=candidate.page_number,
                hierarchy_path=candidate.hierarchy_path,
                subject=candidate.subject,
            )
        )

    return "\n\n".join(blocks), sources


def generate_answer(query: str, candidates: list[Candidate]) -> Answer:
    """Genere la reponse a partir des extraits retenus.

    Args:
        query: Question de l'utilisateur
        candidates: Candidats retenus apres reranking

    Returns:
        La reponse et ses sources

    Raises:
        RuntimeError: Si la cle Groq est absente
    """
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY absente : renseignez-la dans .env pour activer la generation."
        )

    context, sources = build_context(candidates, settings.max_context_chars)
    if not context:
        return Answer(
            text="Aucun extrait pertinent n'a ete trouve pour repondre a cette question.",
            sources=[],
        )

    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    completion = client.chat.completions.create(
        model=settings.groq_model,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(context, query)},
        ],
    )
    text = completion.choices[0].message.content.strip()
    logger.info("Reponse generee ({} caracteres, {} sources)", len(text), len(sources))
    return Answer(text=text, sources=sources)
