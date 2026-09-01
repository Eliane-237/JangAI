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

import time
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
            # `get_sentence_embedding_dimension` est deprecie ; on prend le nom
            # recent s'il existe, l'ancien sinon.
            if hasattr(self._model, "get_embedding_dimension"):
                dimension = self._model.get_embedding_dimension()
            else:
                dimension = self._model.get_sentence_embedding_dimension()
            if dimension != self._settings.embedding_dim:
                raise ValueError(
                    f"Le modele produit {dimension} dimensions alors que la "
                    f"configuration en declare {self._settings.embedding_dim}. "
                    f"Alignez embedding_dim et le schema SQL."
                )
            logger.info("Modele charge ({} dimensions)", dimension)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode un lot, sans journalisation."""
        vectors = self.model.encode(
            texts,
            batch_size=self._settings.embedding_batch_size,
            normalize_embeddings=self._settings.embedding_normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def generate_embeddings(
        self, texts: list[str], describe: str = "textes"
    ) -> np.ndarray:
        """Encode une liste de textes bruts, avec journalisation detaillee.

        Le travail est fait lot par lot pour rendre la vectorisation visible :
        chaque lot est trace (taille, duree, debit), et les proprietes des
        vecteurs produits (dimension, norme, plage de valeurs) sont resumees.
        Un texte isole (cas d'une requete) est encode silencieusement.

        Args:
            texts: Textes a vectoriser
            describe: Libelle des elements, pour les logs ("chunks", "textes")

        Returns:
            Matrice de vecteurs, une ligne par texte
        """
        if not texts:
            return np.empty((0, self._settings.embedding_dim), dtype=np.float32)

        # Cas d'une requete unique : pas de bruit dans les logs.
        if len(texts) == 1:
            vector = self._encode(texts)
            logger.debug(
                "Requete vectorisee | dim {} | norme {:.3f}",
                vector.shape[1], float(np.linalg.norm(vector[0])),
            )
            return vector

        settings = self._settings
        batch_size = settings.embedding_batch_size
        total = len(texts)
        batches = (total + batch_size - 1) // batch_size
        logger.info(
            "Vectorisation de {} {} | modele {} | device {} | batch {} | normalize {}",
            total, describe, settings.embedding_model, settings.embedding_device,
            batch_size, settings.embedding_normalize,
        )

        parts: list[np.ndarray] = []
        done = 0
        started = time.perf_counter()
        for index in range(batches):
            start = index * batch_size
            batch = texts[start : start + batch_size]
            t0 = time.perf_counter()
            parts.append(self._encode(batch))
            done += len(batch)
            elapsed = time.perf_counter() - started
            logger.info(
                "  lot {}/{} : {} vecteurs en {:.1f}s  (cumul {}/{}, ~{:.1f} chunks/s)",
                index + 1, batches, len(batch), time.perf_counter() - t0,
                done, total, done / elapsed if elapsed else 0.0,
            )

        vectors = np.vstack(parts).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        logger.info(
            "{} vecteurs | dim {} | norme moy {:.3f} (min {:.3f}, max {:.3f}) | "
            "valeurs [{:.3f}, {:.3f}] | {:.1f}s",
            len(vectors), vectors.shape[1], float(norms.mean()),
            float(norms.min()), float(norms.max()),
            float(vectors.min()), float(vectors.max()),
            time.perf_counter() - started,
        )
        logger.debug(
            "Echantillon vecteur[0][:8] = {}", np.round(vectors[0][:8], 4).tolist()
        )
        return vectors

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
        return self.generate_embeddings(
            [d.contextualized_text() for d in documents], describe="chunks"
        )

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