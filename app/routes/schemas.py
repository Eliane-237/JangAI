"""
Schemas d'entree/sortie partages par les routes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Requete commune a /ask et /search."""

    question: str = Field(..., min_length=3, description="Question de l'utilisateur")
    top_k: int | None = Field(default=None, description="Nombre d'extraits transmis")
    subject: str | None = Field(default=None, description="Filtre matiere")
    level: str | None = Field(default=None, description="Filtre niveau")
    track: str | None = Field(default=None, description="Filtre serie")

    def filters(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "subject": self.subject,
                "level": self.level,
                "track": self.track,
            }.items()
            if value
        }
