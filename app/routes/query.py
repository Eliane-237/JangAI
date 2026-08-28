"""
Route /ask — reponse complete generee, avec sources.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.routes.schemas import QueryRequest
from app.services.rag_pipeline import answer_question

router = APIRouter(tags=["query"])


@router.post("/ask")
def ask(request: QueryRequest) -> dict:
    """Repond a une question a partir du corpus, avec citations."""
    try:
        answer = answer_question(
            request.question, top_k=request.top_k, filters=request.filters()
        )
    except RuntimeError as exc:
        # Cle Groq absente : configuration incomplete, pas une erreur serveur.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Echec de la generation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return answer.to_dict()
