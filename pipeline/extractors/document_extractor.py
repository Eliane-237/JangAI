"""
Extraction — reconstruction de la structure du document.

Responsabilite unique : produire un `DoclingDocument` complet a partir d'un PDF.

Docling reconstruit la structure de tout le document (titres, listes, tableaux,
ordre de lecture). L'OCR est ACTIF, avec Tesseract comme moteur (jamais RapidOCR)
et TableFormer branche : sur une page scannee, Docling OCRise avec Tesseract PUIS
reconstruit la structure des tableaux — les colonnes d'un tableau scanne sont donc
preservees, la ou un OCR a plat les entremelerait. Sur une page native, Docling
saute l'OCR et garde la couche texte existante.

Filet de securite : si une page manifestement scannee ressort sans aucun texte
apres le tour Docling, elle est rattrapee par le module `ocr.py` (Tesseract +
pretraitement OpenCV) et son texte est reinjecte. Le document rendu est homogene :
le decoupeur, en aval, n'a plus a savoir d'ou vient chaque page.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pymupdf
from loguru import logger

from app.config import THRESHOLDS, get_settings
from app.models import Reliability
from pipeline.extractors.ocr import (
    build_docling_ocr_options,
    extract_text_from_pdf_page,
    is_likely_scanned,
    tesseract_available,
)

# Tesseract tente une detection d'orientation (OSD) sur de petites vignettes,
# echoue, puis poursuit l'OCR normalement (comportement documente dans Docling).
# Docling journalise cet echec en ERROR alors qu'il est sans consequence : on le
# filtre precisement, sans masquer les autres erreurs (dont un vrai echec d'OCR).
class _DropOSDFailures(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            return "OSD failed" not in record.getMessage()
        except Exception:
            return True


for _name in (
    "docling.models.stages.ocr.tesseract_ocr_cli_model",
    "docling.models.stages.ocr.tesseract_ocr_model",
):
    logging.getLogger(_name).addFilter(_DropOSDFailures())


# ======================================================================
# Convertisseur Docling (charge une seule fois par processus)
# ======================================================================


@lru_cache(maxsize=2)
def get_converter(with_ocr: bool):
    """Convertisseur Docling, avec ou sans OCR selon le document.

    Deux instances mises en cache :
        with_ocr=False  documents entierement natifs. Tesseract ne tourne
                        jamais — pas meme sur les images. Plus rapide, sortie
                        propre. TableFormer reste actif pour les tableaux natifs.
        with_ocr=True   documents comportant des pages scannees. Tesseract
                        (via `ocr.py`) OCRise ces pages, puis TableFormer en
                        reconstruit les tableaux.

    Args:
        with_ocr: Activer l'OCR Tesseract

    Returns:
        Le `DocumentConverter` configure
    """
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    settings = get_settings()
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = with_ocr
    if with_ocr:
        pipeline_options.ocr_options = build_docling_ocr_options()
    pipeline_options.do_table_structure = settings.docling_table_structure
    pipeline_options.table_structure_options.do_cell_matching = True
    # Les modeles Docling (mise en page + TableFormer) sur GPU si disponible.
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=settings.docling_device, num_threads=settings.docling_num_threads
    )

    logger.info(
        "Chargement du convertisseur Docling (TableFormer{}, device {})",
        " + OCR Tesseract" if with_ocr else ", sans OCR — document natif",
        settings.docling_device,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


# ======================================================================
# Signaux de page (PyMuPDF) : reperer les pages scannees
# ======================================================================


def _page_needs_ocr(page: pymupdf.Page) -> tuple[bool, bool]:
    """Une page est-elle depourvue de couche texte exploitable ?

    Sert a marquer la fiabilite des chunks issus de pages scannees, et a cibler
    le filet de securite. Signaux physiques : caracteres natifs, part d'image,
    densite de traces vectoriels.

    Args:
        page: Page PyMuPDF

    Returns:
        Tuple (page scannee, ressemble a un tableau)
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


def _detect_scanned_pages(path: Path) -> dict[int, bool]:
    """Pages scannees du document, avec l'indice "ressemble a un tableau".

    Returns:
        Dictionnaire {numero de page (1-base) -> is_table}
    """
    scanned: dict[int, bool] = {}
    with pymupdf.open(path) as pdf:
        for page in pdf:
            needs, is_table = _page_needs_ocr(page)
            if needs:
                scanned[page.number + 1] = is_table
    return scanned


# ======================================================================
# Lecture d'un DoclingDocument
# ======================================================================


def _page_of(item) -> int | None:
    """Numero de page (1-base) d'un element Docling, lu dans sa provenance."""
    provenance = getattr(item, "prov", None) or []
    return getattr(provenance[0], "page_no", None) if provenance else None


def _pages_with_text(doc) -> set[int]:
    """Pages pour lesquelles Docling a produit au moins un element textuel."""
    pages: set[int] = set()
    for item, _level in doc.iterate_items():
        if (getattr(item, "text", "") or "").strip():
            page = _page_of(item)
            if page is not None:
                pages.add(page)
    return pages


# ======================================================================
# Filet de securite : OCR ocr.py sur les pages scannees restees muettes
# ======================================================================


def _inject_ocr_text(doc, page_no: int, text: str) -> bool:
    """Ajoute au `DoclingDocument` le texte OCR d'une page, sur sa page."""
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


def _rescue_empty_scanned_pages(
    path: Path, doc, scanned: dict[int, bool]
) -> set[int]:
    """Rattrape par `ocr.py` les pages scannees que Docling n'a pas su lire.

    Docling gere deja l'OCR (Tesseract + TableFormer) ; ce filet ne se declenche
    que pour les pages scannees restees SANS aucun texte apres sa passe, en
    s'appuyant sur le pretraitement OpenCV de `ocr.py`.

    Args:
        path: Chemin du PDF source
        doc: DoclingDocument a completer
        scanned: Pages scannees detectees (page -> is_table)

    Returns:
        Pages effectivement rattrapees
    """
    empty = [p for p in scanned if p not in _pages_with_text(doc)]
    if not empty or not tesseract_available():
        return set()

    logger.info("Filet de securite ocr.py : {} page(s) scannee(s) muette(s) {}",
                len(empty), sorted(empty))
    rescued: set[int] = set()
    with pymupdf.open(path) as pdf:
        for page_no in sorted(empty):
            text, reliability, _score = extract_text_from_pdf_page(
                pdf[page_no - 1], path, is_table=scanned[page_no]
            )
            if not text:
                logger.warning("Aucun texte OCR obtenu p.{}", page_no)
                continue
            if reliability is Reliability.LOW:
                logger.warning("OCR peu fiable p.{}", page_no)
            if _inject_ocr_text(doc, page_no, text):
                rescued.add(page_no)
    return rescued


# ======================================================================
# Point d'entree
# ======================================================================


def extract_document(path: Path) -> tuple[object, set[int]]:
    """Convertit un PDF en `DoclingDocument` structure, OCR compris.

    Args:
        path: Chemin du fichier PDF

    Returns:
        Tuple (DoclingDocument complet, numeros des pages scannees)
    """
    # Pre-scan PyMuPDF (aucun modele) : l'OCR n'est arme que si le document
    # contient reellement des pages scannees. Un document natif n'est donc
    # jamais touche par Tesseract.
    scanned = _detect_scanned_pages(path)
    logger.info(
        "Conversion Docling de {} ({})",
        path.name,
        f"{len(scanned)} page(s) scannee(s) -> OCR" if scanned else "natif, sans OCR",
    )

    doc = get_converter(bool(scanned)).convert(str(path)).document

    if scanned:
        _rescue_empty_scanned_pages(path, doc, scanned)

    return doc, set(scanned)


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
