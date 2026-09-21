"""
Generation de la reponse par un LLM (Groq).

Le modele ne repond QUE sur la base des extraits fournis, et cite ses sources.
Cette contrainte est ce qui distingue une reponse tracable d'une hallucination :
chaque affirmation renvoie a un extrait numerote, rattache a un fichier et une
page.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from loguru import logger

from app.config import get_settings
from app.prompts.templates import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.search import Candidate


@lru_cache(maxsize=1)
def _client():
    """Client Groq unique par processus."""
    from groq import Groq

    return Groq(api_key=get_settings().groq_api_key)


def complete(
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Un aller-retour au LLM Groq, reutilisable (generation, reformulation...).

    Args:
        system: Message systeme
        user: Message utilisateur
        temperature: Temperature (defaut : celle de la config)
        max_tokens: Plafond de tokens (defaut : celui de la config)

    Returns:
        Le texte de la reponse

    Raises:
        RuntimeError: Si la cle Groq est absente
    """
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY absente : renseignez-la dans .env pour activer le LLM."
        )
    completion = _client().chat.completions.create(
        model=settings.groq_model,
        temperature=settings.groq_temperature if temperature is None else temperature,
        max_tokens=settings.groq_max_tokens if max_tokens is None else max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


def chat_with_tools(
    messages: list[dict],
    tools: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tool_choice: str = "auto",
):
    """Un tour de LLM AVEC outils : renvoie le message brut (contenu OU appels).

    Contrairement a `complete`, ce helper expose les `tool_calls` : c'est le
    LLM qui decide d'appeler un outil (ex. la recherche) ou de repondre
    directement. La boucle agent <-> outils est orchestree par l'appelant.

    Args:
        messages: Historique complet du tour (system, user, assistant, tool...)
        tools: Schemas des outils au format OpenAI/Groq
        tool_choice: "auto" (le LLM decide) ou "none" (reponse forcee, sans outil)

    Returns:
        L'objet message de la reponse (`.content` et/ou `.tool_calls`)

    Raises:
        RuntimeError: Si la cle Groq est absente
    """
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY absente : renseignez-la dans .env pour activer l'agent."
        )
    params: dict = {
        "model": settings.groq_model,
        "temperature": settings.groq_temperature if temperature is None else temperature,
        "max_tokens": settings.groq_max_tokens if max_tokens is None else max_tokens,
        "messages": messages,
    }
    # tool_choice="none" : on N'ENVOIE PAS les outils. Certains modeles (gpt-oss)
    # ignorent "none" et appellent quand meme un outil, ce que Groq rejette
    # (400). Retirer les outils rend l'appel impossible : le modele conclut.
    if tool_choice != "none":
        params["tools"] = tools
        params["tool_choice"] = tool_choice
    completion = _client().chat.completions.create(**params)
    return completion.choices[0].message


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


def build_context(
    candidates: list[Candidate], max_chars: int, start_index: int = 1
) -> tuple[str, list[Source]]:
    """Assemble le contexte numerote et la liste des sources.

    Args:
        candidates: Candidats retenus, deja ordonnes
        max_chars: Budget de caracteres pour le contexte
        start_index: Numero du premier extrait. Permet a l'agent d'enchainer
            plusieurs recherches en gardant des index de sources DISTINCTS
            (la 2e recherche continue la numerotation de la 1re, sans collision).

    Returns:
        Tuple (contexte textuel, sources)
    """
    blocks: list[str] = []
    sources: list[Source] = []
    used = 0

    for index, candidate in enumerate(candidates, start=start_index):
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


def generate_answer(
    query: str,
    candidates: list[Candidate],
    history: list[dict] | None = None,
) -> Answer:
    """Genere la reponse a partir des extraits retenus.

    Args:
        query: Question de l'utilisateur
        candidates: Candidats retenus apres reranking
        history: Tours precedents [{question, answer}], pour la coherence
            conversationnelle. Les extraits restent la seule source de verite ;
            l'historique ne sert qu'a lever les references ("ca", "et pour...").

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

    # Historique compact avant la question courante, pour que le modele
    # comprenne les references sans perdre l'ancrage sur les extraits.
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if turn.get("question"):
            messages.append({"role": "user", "content": turn["question"]})
        if turn.get("answer"):
            messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": build_user_prompt(context, query)})

    completion = _client().chat.completions.create(
        model=settings.groq_model,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
        messages=messages,
    )
    text = (completion.choices[0].message.content or "").strip()
    logger.info("Reponse generee ({} caracteres, {} sources)", len(text), len(sources))
    return Answer(text=text, sources=sources)
