"""
Reconnaissance optique de caracteres.

Intervient sur les pages dont l'analyse structurelle a etabli qu'elles ne
comportent pas de couche texte exploitable : pages scannees, ou pages dont
le texte a ete converti en courbes vectorielles.

Chaine : rendu haute resolution -> pretraitement image -> Tesseract ->
post-traitement du texte.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pymupdf
from loguru import logger

from app.config import OCR_CACHE_DIR, THRESHOLDS, get_settings
from app.models import Reliability

try:
    import cv2
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    OCR_AVAILABLE = False
    logger.warning("Dependances OCR indisponibles ({}). L'OCR sera ignore.", exc)


# ======================================================================
# Detection de langue
# ======================================================================

_FR_WORDS = {
    "le", "la", "les", "des", "une", "dans", "pour", "sur", "avec", "est",
    "sont", "que", "qui", "aux", "par", "plus", "cette", "leur", "ses",
}
_EN_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "will",
    "are", "was", "their", "which", "they", "been", "would", "about",
}


def analyze_document_language(text: str) -> str:
    """Determine la langue dominante d'un texte.

    Methode volontairement simple et sans dependance : le comptage de mots
    outils discrimine nettement le francais de l'anglais des quelques
    dizaines de mots.

    Args:
        text: Texte a analyser

    Returns:
        Code langue "fr" ou "en"
    """
    words = re.findall(r"[a-zà-ÿ]+", text.lower())
    if len(words) < 20:
        return "fr"
    counts = Counter(words)
    fr_score = sum(counts[w] for w in _FR_WORDS)
    en_score = sum(counts[w] for w in _EN_WORDS)
    return "en" if en_score > fr_score else "fr"


# ======================================================================
# Pretraitement de l'image
# ======================================================================


def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """Ameliore la lisibilite d'une page avant reconnaissance.

    Quatre operations, dans cet ordre :
        1. niveaux de gris — la couleur n'apporte rien au texte ;
        2. debruitage doux — supprime le grain sans eroder les jambages ;
        3. seuillage adaptatif — resiste aux eclairages inegaux des scans ;
        4. dilatation legere — reconnecte les traits interrompus, frequents
           sur du texte vectorise rendu a resolution moderee.

    Args:
        image: Image source au format tableau numpy

    Returns:
        Image binarisee prete pour Tesseract
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    denoised = cv2.fastNlMeansDenoising(
        gray, None, h=7, templateWindowSize=7, searchWindowSize=21
    )
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    kernel = np.ones((1, 1), np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)


# ======================================================================
# Post-traitement du texte
# ======================================================================

# Confusions recurrentes de Tesseract sur du texte francais.
_CORRECTIONS = (
    (re.compile(r"[ \t]{2,}"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
    (re.compile(r"(?<=[a-zà-ÿ])\|(?=[a-zà-ÿ])"), "l"),   # | lu pour l
    (re.compile(r"(?<=\d)[oO](?=\d)"), "0"),              # O lu pour 0
    (re.compile(r"(?<=[A-Za-z])0(?=[a-z])"), "o"),        # 0 lu pour o
    (re.compile(r"[«»\u201c\u201d]"), '"'),
    (re.compile(r"[\u2018\u2019´`]"), "'"),
)


def post_process_ocr_text(text: str) -> str:
    """Nettoie la sortie brute de Tesseract.

    Args:
        text: Texte reconnu, non corrige

    Returns:
        Texte nettoye
    """
    result = text
    for pattern, replacement in _CORRECTIONS:
        result = pattern.sub(replacement, result)
    lines = [l.rstrip() for l in result.splitlines()]
    # Une ligne d'un seul caractere non alphanumerique est du bruit.
    lines = [l for l in lines if len(l.strip()) != 1 or l.strip().isalnum()]
    return "\n".join(lines).strip()


def estimate_ocr_reliability(data: dict) -> tuple[float, Reliability]:
    """Evalue la confiance de la reconnaissance.

    Tesseract attribue une confiance a chaque mot. La moyenne sur les mots
    reellement reconnus donne une mesure exploitable pour ponderer le chunk
    au reranking. Une reconnaissance, meme excellente, ne vaut jamais une
    extraction native : le meilleur niveau atteignable est MEDIUM.

    Args:
        data: Dictionnaire renvoye par `image_to_data`

    Returns:
        Tuple (score entre 0 et 1, niveau de fiabilite)
    """
    scores = [
        int(c)
        for c, word in zip(data.get("conf", []), data.get("text", []))
        if str(c).lstrip("-").isdigit() and int(c) >= 0 and word.strip()
    ]
    if not scores:
        return 0.0, Reliability.LOW

    average = sum(scores) / len(scores) / 100.0
    level = (
        Reliability.MEDIUM
        if average >= THRESHOLDS.min_ocr_confidence
        else Reliability.LOW
    )
    return average, level


# ======================================================================
# Cache
# ======================================================================


def _cache_path(source: Path, page_number: int, psm: int) -> Path:
    """Chemin du fichier de cache pour une page donnee.

    L'empreinte inclut la taille et la date du fichier source : modifier le
    PDF invalide automatiquement le cache.
    """
    stat = source.stat()
    seed = f"{source.name}:{stat.st_size}:{int(stat.st_mtime)}:{page_number}:{psm}"
    digest = hashlib.sha256(seed.encode()).hexdigest()[:20]
    return OCR_CACHE_DIR / f"{source.stem}_p{page_number}_{digest}.txt"


# ======================================================================
# Point d'entree
# ======================================================================


def is_likely_scanned(char_count: int, vector_paths: int, image_ratio: float) -> bool:
    """Indique si une page requiert vraisemblablement l'OCR.

    Reprend les memes signaux que `classify_layout`, sous une forme
    utilisable hors du contexte d'analyse complete.

    Args:
        char_count: Nombre de caracteres extraits nativement
        vector_paths: Nombre de traces vectoriels
        image_ratio: Part de la page couverte par des images

    Returns:
        True si la page doit passer par l'OCR
    """
    if char_count >= THRESHOLDS.min_text_chars:
        return False
    return (
        image_ratio >= THRESHOLDS.min_image_ratio_scanned
        or vector_paths >= THRESHOLDS.min_vector_paths
    )


def extract_text_from_pdf_page(
    page: pymupdf.Page, source_path: Path, is_table: bool = False
) -> tuple[str, Reliability, float]:
    """Reconnait le texte d'une page par OCR.

    Args:
        page: Page PyMuPDF a traiter
        source_path: Chemin du PDF source, pour la cle de cache
        is_table: Si vrai, utilise le mode de segmentation adapte aux
            grilles (psm 6), nettement meilleur que la segmentation
            automatique sur du contenu tabulaire

    Returns:
        Tuple (texte reconnu, fiabilite, score de confiance)
    """
    settings = get_settings()
    if not OCR_AVAILABLE:
        return "", Reliability.LOW, 0.0

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    psm = settings.ocr_psm_table if is_table else settings.ocr_psm_prose

    cache_file = _cache_path(source_path, page.number + 1, psm)
    if settings.ocr_cache_enabled and cache_file.exists():
        logger.debug("OCR p.{} servi depuis le cache", page.number + 1)
        return cache_file.read_text(encoding="utf-8"), Reliability.MEDIUM, 0.85

    try:
        # `get_pixmap` applique la rotation declaree de la page : le texte
        # arrive donc a l'endroit, sans redressement manuel.
        matrix = pymupdf.Matrix(settings.ocr_dpi / 72, settings.ocr_dpi / 72)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        if pixmap.n == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif pixmap.n == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        prepared = preprocess_image_for_ocr(image)
        config = f"--oem {settings.ocr_oem} --psm {psm}"

        data = pytesseract.image_to_data(
            Image.fromarray(prepared),
            lang=settings.ocr_languages,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        text = post_process_ocr_text(
            pytesseract.image_to_string(
                Image.fromarray(prepared),
                lang=settings.ocr_languages,
                config=config,
            )
        )
        score, reliability = estimate_ocr_reliability(data)

        if settings.ocr_cache_enabled and text:
            cache_file.write_text(text, encoding="utf-8")

        logger.debug(
            "OCR p.{} : {} caracteres, confiance {:.0%}",
            page.number + 1, len(text), score,
        )
        return text, reliability, score

    except Exception as exc:
        logger.error("OCR en echec p.{} : {}", page.number + 1, exc)
        return "", Reliability.LOW, 0.0