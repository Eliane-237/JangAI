"""
Extraction — reconstruction de la structure du document.

Responsabilite unique : produire un `DoclingDocument` complet a partir d'un
PDF.

Docling reconstruit la structure de tout le document (titres, listes,
tableaux, ordre de lecture). Il n'execute AUCUN OCR interne (`do_ocr=False`) :
les pages depourvues de couche texte — scannees ou vectorisees — sont
reconnues par notre propre module `ocr.py` (Tesseract + pretraitement OpenCV),
puis leur texte est reinjecte dans le `DoclingDocument`. Le document rendu est
donc homogene : le decoupeur, en aval, n'a plus a savoir d'ou vient chaque
page.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pymupdf
from loguru import logger

from app.config import THRESHOLDS, get_settings
from app.models import Reliability
from pipeline.extractors.ocr import (
    extract_text_from_pdf_page,
    is_likely_scanned,
)


# ======================================================================
# Convertisseur Docling (charge une seule fois par processus)
# ======================================================================


@lru_cache(maxsize=1)
def get_converter():
    """Convertisseur Docling : structure seule, sans OCR interne.

    Docling ne fait ici que la mise en page et la reconnaissance de tableaux
    (TableFormer). L'OCR est deliberement desactive : il est confie a `ocr.py`.

    Returns:
        Le `DocumentConverter` configure
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    settings = get_settings()
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False                       # OCR delegue a ocr.py
    pipeline_options.do_table_structure = settings.docling_table_structure
    pipeline_options.table_structure_options.do_cell_matching = True

    logger.info("Chargement du convertisseur Docling (TableFormer, OCR interne desactive)")
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


# ======================================================================
# Signaux de page (PyMuPDF) pour reperer les pages a OCRiser
# ======================================================================


def _page_needs_ocr(page: pymupdf.Page) -> tuple[bool, bool]:
    """Une page est-elle depourvue de couche texte exploitable ?

    Reprend les memes signaux physiques que l'ancienne analyse de mise en
    page, mais reduits au strict necessaire : nombre de caracteres natifs,
    part d'image, densite de traces vectoriels.

    Args:
        page: Page PyMuPDF

    Returns:
        Tuple (besoin d'OCR, ressemble a un tableau)
    """
    text = page.get_text("text")
    surface = (page.rect.width * page.rect.height) or 1.0

    image_surface = 0.0
    for image in page.get_images(full=True):
        for rect in page.get_image_rects(image[0]):
            image_surface += rect.width * rect.height

    vector_paths = len(page.get_drawings())
    needs = is_likely_scanned(
        char_count=len(text.strip()),
        vector_paths=vector_paths,
        image_ratio=image_surface / surface,
    )
    is_table = vector_paths >= THRESHOLDS.min_vector_paths
    return needs, is_table


# ======================================================================
# Injection du texte OCR dans le DoclingDocument
# ======================================================================


def _inject_ocr_text(doc, page_no: int, text: str) -> bool:
    """Ajoute le texte reconnu d'une page scannee au `DoclingDocument`.

    Le texte est insere comme un element `text` porte par la page concernee,
    de sorte que le decoupeur le traite ensuite comme n'importe quel autre
    contenu natif.

    Args:
        doc: DoclingDocument a completer
        page_no: Numero de page (1-base)
        text: Texte reconnu par l'OCR

    Returns:
        True si l'insertion a reussi
    """
    try:
        from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel
        from docling_core.types.doc.document import ProvenanceItem

        page = doc.pages.get(page_no) if hasattr(doc.pages, "get") else None
        size = getattr(page, "size", None)
        width = float(getattr(size, "width", 0.0)) or 595.0
        height = float(getattr(size, "height", 0.0)) or 842.0

        prov = ProvenanceItem(
            page_no=page_no,
            bbox=BoundingBox(
                l=0.0, t=0.0, r=width, b=height, coord_origin=CoordOrigin.TOPLEFT
            ),
            charspan=(0, len(text)),
        )
        doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)
        return True
    except Exception as exc:  # pragma: no cover - depend de la version Docling
        logger.warning("Injection du texte OCR impossible p.{} : {}", page_no, exc)
        return False


def _ocr_scanned_pages(path: Path, doc) -> set[int]:
    """Reconnait par `ocr.py` les pages sans couche texte et les injecte.

    Args:
        path: Chemin du PDF source
        doc: DoclingDocument a completer

    Returns:
        Ensemble des numeros de pages effectivement OCRisees
    """
    ocr_pages: set[int] = set()

    with pymupdf.open(path) as pdf:
        candidates = []
        for page in pdf:
            needs, is_table = _page_needs_ocr(page)
            if needs:
                candidates.append((page.number + 1, is_table))

        if not candidates:
            return ocr_pages

        logger.info("OCR (ocr.py / Tesseract) sur {} page(s) scannee(s)", len(candidates))
        for page_no, is_table in candidates:
            text, reliability, _score = extract_text_from_pdf_page(
                pdf[page_no - 1], path, is_table=is_table
            )
            if not text:
                logger.warning("Aucun texte OCR obtenu p.{}", page_no)
                continue
            if reliability is Reliability.LOW:
                logger.warning("OCR peu fiable p.{}", page_no)
            if _inject_ocr_text(doc, page_no, text):
                ocr_pages.add(page_no)

    return ocr_pages


# ======================================================================
# Point d'entree
# ======================================================================


def extract_document(path: Path) -> tuple[object, set[int]]:
    """Convertit un PDF en `DoclingDocument`, pages scannees OCRisees comprises.

    Args:
        path: Chemin du fichier PDF

    Returns:
        Tuple (DoclingDocument complet, numeros des pages OCRisees)
    """
    logger.info("Conversion Docling de {}", path.name)
    doc = get_converter().convert(str(path)).document
    ocr_pages = _ocr_scanned_pages(path, doc)
    return doc, ocr_pages


def _page_of(item) -> int | None:
    """Numero de page (1-base) d'un element Docling, lu dans sa provenance."""
    provenance = getattr(item, "prov", None) or []
    return getattr(provenance[0], "page_no", None) if provenance else None


def document_header_text(doc, max_page: int = 6) -> str:
    """Texte des premieres pages, pour l'identification du document.

    Args:
        doc: DoclingDocument
        max_page: Derniere page consideree comme en-tete

    Returns:
        Texte concatene des premieres pages
    """
    parts: list[str] = []
    for item, _level in doc.iterate_items():
        page = _page_of(item)
        text = (getattr(item, "text", "") or "").strip()
        if text and (page is None or page <= max_page):
            parts.append(text)
    return "\n".join(parts)
