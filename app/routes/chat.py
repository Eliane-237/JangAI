"""
Route /chat — reponse via l'agent LangGraph.

Contrairement a /ask (pipeline lineaire fige), /chat passe par le GRAPHE de
l'agent : routage des facettes -> recherche -> generation (et, en phase 2, la
boucle auto-corrective). L'agent decide lui-meme des filtres a appliquer.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from app.agent.graph import answer as run_agent
from app.config import get_settings
from app.routes.schemas import ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Repond a une question via l'agent (memoire + routage + recherche + gen.).

    Renvoie un `thread_id` : renvoyez-le au tour suivant pour conserver le fil
    de la conversation (l'agent resout alors « tout ca », « et pour... », etc.).
    """
    try:
        return run_agent(request.question, request.thread_id)
    except RuntimeError as exc:
        # Cle Groq absente : configuration incomplete, pas une erreur serveur.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Echec de l'agent")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat/voice")
async def chat_voice(
    file: UploadFile = File(...),
    thread_id: str | None = Form(default=None),
) -> dict:
    """Pose une question A LA VOIX : audio -> transcription -> reponse de l'agent.

    Enchaine /transcribe et /chat en un seul appel. Renvoie en plus la
    `transcription` (ce que Jang a entendu) pour que le client puisse l'afficher.
    `thread_id` conserve la memoire multi-tour comme sur /chat : vide au premier
    message, renvoyez celui recu ensuite.
    """
    # Import differe : evite de charger faster-whisper quand seule la voie texte
    # est utilisee.
    from app.services.transcription import transcribe

    settings = get_settings()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Fichier audio vide.")
    if len(content) > settings.stt_max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Audio trop volumineux (> {settings.stt_max_upload_mb} Mo).",
        )

    try:
        heard = transcribe(content)
    except Exception as exc:  # pragma: no cover
        logger.exception("Echec de la transcription")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    question = heard["text"].strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="Aucune parole detectee dans l'audio.",
        )

    try:
        result = run_agent(question, thread_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Echec de l'agent")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # On expose ce qui a ete entendu, en plus de la reponse de l'agent.
    return {"transcription": question, **result}
