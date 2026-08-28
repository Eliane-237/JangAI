"""
Orchestration de l'ingestion.

Chaine complete, moteur Docling unique :
    convertir -> identifier -> decouper (HybridChunker) -> dedupliquer ->
    vectoriser -> inserer -> index HNSW.

Utilisation :
    python -m pipeline.ingest                     # tout data/raw
    python -m pipeline.ingest --file X.pdf        # un seul fichier
    python -m pipeline.ingest --skip-index        # differer l'index
    python -m pipeline.ingest --dry-run           # sans ecrire en base
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from loguru import logger

from app.config import RAW_DIR, check_configuration, get_settings
from app.db import check_connection, get_statistics
from app.models import generate_document_id
from pipeline.chunking import chunk_document
from pipeline.deduplicator import deduplicate
from pipeline.embedding import get_embedding_service
from pipeline.extractors.document_extractor import (
    document_header_text,
    extract_document,
)
from pipeline.indexing import (
    get_quality_report,
    insert_documents_with_metadata,
    rebuild_index,
    save_document,
)
from pipeline.preprocessor import identify_document


class IngestionPipeline:
    """Pipeline d'ingestion d'un corpus de documents PDF."""

    def __init__(self, dry_run: bool = False, fuzzy_dedup: bool = True) -> None:
        self.dry_run = dry_run
        self.fuzzy_dedup = fuzzy_dedup
        self.failures: list[str] = []

    # ------------------------------------------------------------------

    def load_documents(self, directory: Path) -> list[Path]:
        """Liste les fichiers PDF disponibles.

        Args:
            directory: Repertoire a explorer

        Returns:
            Chemins des fichiers PDF trouves

        Raises:
            FileNotFoundError: Si le repertoire n'existe pas
        """
        if not directory.exists():
            raise FileNotFoundError(f"Repertoire introuvable : {directory}")
        files = sorted(directory.glob("*.pdf"))
        logger.info("{} fichier(s) PDF trouve(s) dans {}", len(files), directory)
        return files

    # ------------------------------------------------------------------

    def process_pdf(self, path: Path) -> int:
        """Traite un PDF de bout en bout.

        Args:
            path: Chemin du fichier

        Returns:
            Nombre de chunks produits
        """
        started = time.perf_counter()
        logger.info("{}", "=" * 62)
        logger.info("Ingestion de {}", path.name)

        # -- 1. Extraction (Docling structure + OCR ocr.py) ------------
        # Docling reconstruit la structure ; les pages scannees sont OCRisees
        # par ocr.py et reinjectees. Le document rendu est homogene.
        doc, ocr_pages = extract_document(path)

        # -- 2. Identification -----------------------------------------
        identity = identify_document(document_header_text(doc), path.name)

        # -- 3. Decoupage (HybridChunker) ------------------------------
        # Le profil du document est renseigne par `chunk_document` a partir
        # des chunks reellement produits.
        document_id = generate_document_id(path.name, "main")
        documents = chunk_document(
            doc, document_id, path.name, identity, ocr_pages=ocr_pages
        )
        if not documents:
            logger.error("Aucun chunk produit pour {}", path.name)
            return 0

        # -- 4. Deduplication -------------------------------------------
        documents = deduplicate(documents, fuzzy=self.fuzzy_dedup)

        if self.dry_run:
            logger.info("Mode a blanc : rien n'est ecrit en base.")
            for preview in documents[:3]:
                logger.info("  {}", preview)
            return len(documents)

        # -- 5. Vectorisation -------------------------------------------
        embeddings = get_embedding_service().embed_documents(documents)
        logger.info(
            "{} vecteurs produits ({} dimensions)", len(embeddings), embeddings.shape[1]
        )

        # -- 6. Insertion -----------------------------------------------
        save_document(
            document_id=document_id,
            source_file=path.name,
            identity=identity,
            page_count=len(getattr(doc, "pages", {}) or {}),
            gaps=[],
            diagnostics={
                "engine": "docling",
                "profile": identity.profile.value,
                "chunk_types": dict(
                    Counter(d.metadata.chunk_type.value for d in documents)
                ),
            },
            sub_document_id="main",
        )
        insert_documents_with_metadata(documents, embeddings)

        logger.info("{} termine en {:.1f}s", path.name, time.perf_counter() - started)
        return len(documents)

    # ------------------------------------------------------------------

    def run(self, files: list[Path], build_index: bool = True) -> None:
        """Ingere une liste de fichiers puis reconstruit l'index.

        Args:
            files: Fichiers a traiter
            build_index: Reconstruire l'index HNSW en fin de traitement
        """
        total = 0
        for path in files:
            try:
                total += self.process_pdf(path)
            except Exception as exc:
                # Un fichier en echec ne doit pas interrompre les autres :
                # mieux vaut un corpus partiel qu'aucun corpus.
                logger.exception("Echec sur {} : {}", path.name, exc)
                self.failures.append(path.name)

        if self.dry_run:
            logger.info("Mode a blanc : {} chunks auraient ete produits.", total)
            return

        if build_index and total:
            rebuild_index()

        self.report()

    def report(self) -> None:
        """Affiche l'etat de la base et les alertes qualite."""
        logger.info("{}", "=" * 62)
        stats = get_statistics()
        logger.info(
            "Base : {} documents, {} chunks, {} vectorises",
            stats["documents"], stats["chunks"], stats["embedded"],
        )
        for subject, count in stats["by_subject"].items():
            logger.info("   {:<24} {:>5} chunks", subject or "?", count)

        if missing := stats["chunks"] - stats["embedded"]:
            logger.warning("{} chunk(s) sans vecteur", missing)

        for row in get_quality_report():
            if row.get("gap_count"):
                logger.warning(
                    "{} : {} lacune(s) de pagination",
                    row["source_file"], row["gap_count"],
                )

        if self.failures:
            logger.error("Fichiers en echec : {}", ", ".join(self.failures))


# ======================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingestion du corpus JangAI")
    parser.add_argument("--file", help="Ingerer un seul fichier")
    parser.add_argument("--dir", default=str(RAW_DIR), help="Repertoire des PDF")
    parser.add_argument(
        "--skip-index", action="store_true", help="Ne pas reconstruire l'index HNSW"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Tout executer sans ecrire en base"
    )
    parser.add_argument(
        "--no-fuzzy", action="store_true", help="Desactiver la deduplication approchee"
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        level=get_settings().log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )

    for warning in check_configuration():
        logger.warning(warning)

    if not args.dry_run and not check_connection():
        logger.error("Base indisponible. Lancez `docker compose up -d` et reessayez.")
        return 1

    pipeline = IngestionPipeline(dry_run=args.dry_run, fuzzy_dedup=not args.no_fuzzy)

    if args.file:
        path = Path(args.file)
        if not path.exists():
            path = Path(args.dir) / args.file
        if not path.exists():
            logger.error("Fichier introuvable : {}", args.file)
            return 1
        files = [path]
    else:
        files = pipeline.load_documents(Path(args.dir))

    if not files:
        logger.error("Aucun PDF a traiter dans {}", args.dir)
        return 1

    pipeline.run(files, build_index=not args.skip_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())