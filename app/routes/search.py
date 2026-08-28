"""
Route /search — candidats apres recherche + reranking, sans generation.

Utile pour inspecter la qualite de la recherche independamment du LLM.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.routes.schemas import QueryRequest
from app.services.rag_pipeline import retrieve

router = APIRouter(tags=["search"])


@router.post("/search")
def search(request: QueryRequest) -> dict:
    """Retourne les extraits les plus pertinents et leurs scores."""
    candidates = retrieve(
        request.question, top_k=request.top_k, filters=request.filters()
    )
    return {
        "question": request.question,
        "results": [
            {
                "score": round(c.rerank_score, 4),
                "vector_score": round(c.vector_score, 4),
                "lexical_score": round(c.lexical_score, 4),
                "source_file": c.source_file,
                "page": c.page_number,
                "section": c.hierarchy_text(),
                "content": c.content,
            }
            for c in candidates
        ],
    }
