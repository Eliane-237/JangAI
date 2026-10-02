"""
Client LLM MODULAIRE, compatible OpenAI.

Point unique d'acces au modele de langage. On passe par l'API « OpenAI-
compatible » (standard de fait) : Groq, mais aussi vLLM, Ollama, LM Studio,
text-generation-inference, ou n'importe quel modele que TU deploies, exposent
cette meme interface.

Changer de modele = changer 3 variables dans `.env`, SANS toucher au code :

    LLM_BASE_URL   l'URL du service (ex. http://localhost:8000/v1 pour le tien)
    LLM_API_KEY    la cle si l'endpoint en exige une (vide pour un modele local)
    LLM_MODEL      le nom du modele a appeler

Tout le reste de l'application (generation, agent, reecriture...) appelle
`chat(...)` ici : il n'y a donc qu'UN endroit a reconfigurer.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import get_settings


def base_url() -> str:
    return get_settings().llm_base_url


def api_key() -> str:
    # Repli sur GROQ_API_KEY pour rester compatible avec les .env existants.
    settings = get_settings()
    return settings.llm_api_key or settings.groq_api_key or ""


def model_name() -> str:
    settings = get_settings()
    return settings.llm_model or settings.groq_model


@lru_cache(maxsize=1)
def get_client():
    """Client OpenAI-compatible unique par processus (reconstruit si config change)."""
    from openai import OpenAI

    # Cle facultative pour un modele local : on passe un placeholder non vide
    # car le client exige une chaine.
    return OpenAI(base_url=base_url(), api_key=api_key() or "no-key-required")


def ensure_configured() -> None:
    """Verifie qu'une cle est presente pour un endpoint DISTANT qui en exige une.

    Un modele local (localhost) n'a pas besoin de cle ; un service distant si.
    """
    url = base_url()
    is_local = "localhost" in url or "127.0.0.1" in url
    if not is_local and not api_key():
        raise RuntimeError(
            "Cle LLM absente : renseignez LLM_API_KEY (ou GROQ_API_KEY) dans .env, "
            "ou pointez LLM_BASE_URL vers votre modele local."
        )


def chat(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Any:
    """Un tour de LLM. Renvoie le message brut (`.content` et/ou `.tool_calls`).

    Args:
        messages: Messages au format OpenAI (system/user/assistant/tool).
        tools: Schemas d'outils (optionnel). Ignores si tool_choice == "none".
        tool_choice: "auto" (le LLM decide) ou "none" (reponse forcee, sans outil).
        temperature / max_tokens: surchargent les valeurs par defaut de la config.
    """
    ensure_configured()
    settings = get_settings()
    params: dict[str, Any] = {
        "model": model_name(),
        "messages": messages,
        "temperature": settings.llm_temperature if temperature is None else temperature,
        "max_tokens": settings.llm_max_tokens if max_tokens is None else max_tokens,
    }
    if tools is not None and tool_choice != "none":
        params["tools"] = tools
        params["tool_choice"] = tool_choice
    return get_client().chat.completions.create(**params).choices[0].message
