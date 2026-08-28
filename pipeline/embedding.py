"""
Vectorisation des chunks et des requetes.

Encapsule Qwen3-Embedding-0.6B derriere une interface minimale. Le modele
est charge une seule fois par processus, et paresseusement : le chargement
coute plusieurs secondes et environ 1,5 Go de memoire.

Point critique, souvent manque : Qwen3 attend une INSTRUCTION du cote
requete uniquement, les documents etant encodes bruts. Appliquer la meme
transformation des deux cotes degrade sensiblement le rappel. D'ou deux
methodes distinctes plutot qu'une fonction generique.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from loguru import logger

from app.config import get_settings
from app.models import Document


class EmbeddingService:
    """Service de vectorisation."""

    def __init__(self) -> None:
        self._model = None
        self._settings = get_settings()

    @property
    def model(self):
        """Modele SentenceTransformer, charge au premier acces.

        Returns:
            L'instance du modele

        Raises:
            ValueError: Si la dimension produite ne correspond pas a celle
                declaree dans la configuration et le schema SQL
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Chargement de {} sur {}",
                self._settings.embedding_model,
                self._settings.embedding_device,
            )
            self._model = SentenceTransformer(
                self._settings.embedding_model,
                device=self._settings.embedding_device,
            )
            dimension = self._model.get_sentence_embedding_dimension()
            if dimension != self._settings.embedding_dim:
                raise ValueError(
                    f"Le modele produit {dimension} dimensions alors que la "
                    f"configuration en declare {self._settings.embedding_dim}. "
                    f"Alignez embedding_dim et le schema SQL."
                )
            logger.info("Modele charge ({} dimensions)", dimension)
        return self._model

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Encode une liste de textes bruts.

        Args:
            texts: Textes a vectoriser

        Returns:
            Matrice de vecteurs, une ligne par texte
        """
        if not texts:
            return np.empty((0, self._settings.embedding_dim), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=self._settings.embedding_batch_size,
            normalize_embeddings=self._settings.embedding_normalize,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_documents(self, documents: list[Document]) -> np.ndarray:
        """Encode des chunks.

        C'est `contextualized_text()` qui est encode, non `page_content` :
        une ligne de tableau isolee est trop pauvre pour produire un
        vecteur discriminant, alors que prefixee de son chemin hierarchique
        elle devient exploitable.

        Args:
            documents: Chunks a vectoriser

        Returns:
            Matrice de vecteurs
        """
        return self.generate_embeddings([d.contextualized_text() for d in documents])

    def embed_query(self, query: str) -> np.ndarray:
        """Encode une requete, precedee de son instruction.

        L'instruction n'est appliquee QU'ICI : c'est la dissymetrie
        attendue par Qwen3 entre requetes et documents.

        Args:
            query: Question de l'utilisateur

        Returns:
            Vecteur de la requete
        """
        text = self._settings.query_instruction + query
        return self.generate_embeddings([text])[0]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Instance unique du service."""
    return EmbeddingService()