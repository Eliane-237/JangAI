"""
Route /transcribe — audio vers texte (speech-to-text local).

Point d'entree VOCAL de JangAI. L'enseignant envoie un fichier audio ; on
renvoie le texte transcrit, que le client renvoie ensuite a /chat. On garde la
voix et l'agent DECOUPLES : ici on ne fait que transcrire, le pipeline de
reponse (LangGraph) reste inchange.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

from app.config import get_settings
from app.services.transcription import transcribe

router = APIRouter(tags=["voice"])


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> dict:
    """Transcrit un fichier audio en texte (faster-whisper, 100% local).

    Formats acceptes : wav, mp3, m4a, ogg... (decodes en interne, sans ffmpeg).
    Renvoie `{text, language, duration}`. Enchainez ensuite sur /chat avec le
    `text` pour obtenir la reponse de l'agent.
    """
    settings = get_settings()
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Fichier audio vide.")
    max_bytes = settings.stt_max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio trop volumineux (> {settings.stt_max_upload_mb} Mo).",
        )

    try:
        result = transcribe(content)
    except Exception as exc:  # pragma: no cover
        logger.exception("Echec de la transcription")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not result["text"]:
        logger.warning("Transcription vide pour {} ({} octets)", file.filename, len(content))
    return result
