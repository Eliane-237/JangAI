"""
Route /documents — telechargement des documents generes (.docx).

Sert les fichiers du dossier `generated/` produits par le sous-agent (epreuves,
corriges). L'acces est limite a ce dossier : le nom est assaini pour empecher
toute remontee de chemin (path traversal).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.document_builder import GENERATED_DIR

router = APIRouter(tags=["documents"])

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@router.get("/documents/{name}")
def download_document(name: str) -> FileResponse:
    """Telecharge un document genere par son nom de fichier."""
    # On ne garde que le nom de base : neutralise « ../ » et chemins absolus.
    safe = Path(name).name
    path = (GENERATED_DIR / safe).resolve()
    if GENERATED_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Document introuvable.")
    return FileResponse(path, media_type=_DOCX_MIME, filename=safe)
