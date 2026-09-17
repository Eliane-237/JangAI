"""
Schemas d'entree/sortie partages par les routes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Valeurs de filtre a ignorer : vides, ou le placeholder "string" que Swagger
# injecte dans les champs optionnels.
_IGNORED_FILTER_VALUES = {"", "string"}


class QueryRequest(BaseModel):
    """Requete commune a /ask et /search."""

    # L'exemple ne contient QUE la question : ainsi le bouton "Try it out" de
    # Swagger ne pre-remplit pas les filtres avec le placeholder "string", qui
    # sinon filtrerait sur une matiere inexistante et ne renverrait rien.
    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Quels sont les objectifs de lecture en francais en terminale ?"
            }
        }
    }

    question: str = Field(..., min_length=3, description="Question de l'utilisateur")
    top_k: int | None = Field(default=None, description="Nombre d'extraits transmis")
    subject: str | None = Field(default=None, description="Filtre matiere (optionnel)")
    level: str | None = Field(default=None, description="Filtre niveau (optionnel)")
    track: str | None = Field(default=None, description="Filtre serie (optionnel)")

    def filters(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "subject": self.subject,
                "level": self.level,
                "track": self.track,
            }.items()
            if value and value.strip().lower() not in _IGNORED_FILTER_VALUES
        }


class ChatRequest(BaseModel):
    """Requete de l'agent (/chat).

    L'agent detecte lui-meme les facettes (matiere, niveau...) dans la
    question : pas de filtres a fournir. `thread_id` servira a la memoire
    multi-tour (phase ulterieure).
    """

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Quels sont les objectifs de lecture en francais en terminale ?"
            }
        }
    }

    question: str = Field(..., min_length=3, description="Question de l'utilisateur")
    thread_id: str | None = Field(
        default=None, description="Identifiant de conversation (memoire, a venir)"
    )
