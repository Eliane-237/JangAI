"""
Nettoyage du texte et identification des documents.

Deux responsabilites complementaires, situees en amont du chunking :

    1. `clean_text` — normalisation du texte extrait, reprise et adaptee
       du `preprocess_text` du projet de reference.
    2. `identify_document` — deduction de la matiere, du niveau, de la
       serie, du cycle, de la langue et de l'annee.

Sur l'identification, le NOM DE FICHIER prime pour la matiere, le contenu
ne servant qu'a le confirmer ou a le completer. Ce choix est empirique :
une page de garde contient quantite de tournures en "programme
d'enseignement" ou "programme de reference" qui ressemblent a l'intitule
sans en etre, alors que le nom de fichier est court et deliberement
descriptif. Rien ici n'est propre a une matiere : la methode est de la
tokenisation et du comptage, pas une table de correspondance.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from app.models import DocumentProfile, normalize_key

# Mots outils, pour trancher la langue sans dependance externe.
_FR_WORDS = {
    "le", "la", "les", "des", "une", "dans", "pour", "sur", "avec", "est",
    "sont", "que", "qui", "aux", "par", "plus", "cette", "leur", "ses",
}
_EN_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "will",
    "are", "was", "their", "which", "they", "been", "would", "about",
}


def analyze_document_language(text: str) -> str:
    """Langue dominante d'un texte, par frequence de mots outils.

    Methode volontairement simple et sans dependance : sur plusieurs dizaines
    de mots, le francais et l'anglais se distinguent nettement.

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
# Nettoyage du texte
# ======================================================================

_CLEANING_RULES = (
    # Sequences de numeros parasites laissees par les sommaires.
    (re.compile(r"(\d+\.?\s+)([\s;,]*\d+\.?\s+)+(?=\w)"), r"\1"),
    (re.compile(r"(\d+\.\s+)(\d+\.\s+)+"), r"\1"),
    # Tirets de liste normalises.
    (re.compile(r"-\s+"), "- "),
    # Caracteres de controle.
    (re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"), ""),
    # Espaces multiples.
    (re.compile(r"[ \t]{2,}"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
)

_QUOTE_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00ab": '"', "\u00bb": '"', "\u2013": "-", "\u2014": "-",
}


def clean_text(text: str) -> str:
    """Nettoie et normalise un texte avant indexation.

    Les sauts de ligne internes aux paragraphes sont preserves : ils
    portent l'information de structure sur laquelle s'appuie la detection
    de blocs.

    Args:
        text: Texte brut extrait

    Returns:
        Texte nettoye
    """
    if not text:
        return ""

    cleaned = text
    for character, replacement in _QUOTE_REPLACEMENTS.items():
        cleaned = cleaned.replace(character, replacement)
    for pattern, replacement in _CLEANING_RULES:
        cleaned = pattern.sub(replacement, cleaned)

    return "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()


def strip_accents(text: str) -> str:
    """Retire les accents d'un texte, pour les comparaisons insensibles."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


# ======================================================================
# Identification du document
# ======================================================================

# Segments de nom de fichier ne designant pas une matiere.
_NEUTRAL_SEGMENTS = re.compile(
    r"(?ix)^(?:programmes?|prog|tles?|terminales?|premieres?|secondes?|"
    r"classes?|cycle|officiel|final|version|v\d+|\d+|ls|l[12]|s[123]a?|"
    r"senegal|men|document|doc|pour|locale|de|du|des|la|le)$"
)

# Tournures d'intitule trop generiques pour designer une matiere.
_INVALID_SUBJECTS = {
    "enseignement", "l_enseignement", "reference", "formation", "base",
    "education", "la_classe", "etude", "l_etude",
}

_TITLE_PATTERN = re.compile(
    r"PROGRAMMES?\s+D[E'\u2019]\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s'\u2019-]{2,45}?)"
    r"(?=\s*(?:DU|DE\s|DES\s|POUR|EN\s|A\s|CLASSE|TERMINALE|PREMIER|SERIE|[-–—.,;:(\n])|$)",
    re.I,
)
_LEVEL_PATTERN = re.compile(
    r"\b(TERMINALES?|PREMIERES?|SECONDES?|CM2|CM1|CE2|CE1|CP|CI)\b", re.I
)
# Serie entre guillemets : forme la plus fiable.
_TRACK_QUOTED = re.compile(r"[«\"\u201c'\u2018]\s*([LSG][0-9]?[A-Z]?)\s*[»\"\u201d'\u2019]")
# Serie annoncee, bornee a des jetons courts pour ne pas deborder.
_TRACK_ANNOUNCED = re.compile(
    r"\bS[EÉ]RIES?\s+([LSG][0-9]?[A-Z]?(?:\s*(?:et|&|,|/)\s*[LSG][0-9]?[A-Z]?){0,3})\b",
    re.I,
)
_TRACK_FILENAME = re.compile(r"(?i)[_\-]((?:L|S|G)[0-9]?|LS)(?:[_\-.]|$)")
_YEAR_PATTERN = re.compile(r"\b(?:19[6-9]\d|20[0-9]\d)\b")
_CYCLE_PATTERN = re.compile(
    r"\b([EÉ]L[EÉ]MENTAIRE|MOYEN|SECONDAIRE|PR[EÉ]SCOLAIRE|SUP[EÉ]RIEUR)\b", re.I
)


@dataclass
class DocumentIdentity:
    """Identite deduite d'un document source."""

    subject: str | None = None
    level: str | None = None
    track: str | None = None
    cycle: str | None = None
    language: str = "fr"
    program_year: int | None = None
    profile: DocumentProfile = DocumentProfile.MIXED
    subject_source: str = "unknown"

    def summary(self) -> str:
        parts = [self.subject or "?", self.level or "", self.track or ""]
        return " ".join(p for p in parts if p).strip()


def extract_subject_from_filename(file_name: str) -> str | None:
    """Extrait la matiere du nom de fichier, par elimination.

    On retire les segments de structure (programme, terminale, serie...) ;
    ce qui reste designe la matiere. "Programme_Maths-LS.pdf" laisse
    "maths", "Programmes_SVT_Tles_LS.pdf" laisse "svt".

    Args:
        file_name: Nom du fichier source

    Returns:
        La matiere normalisee, ou None
    """
    segments = [s for s in re.split(r"[_\-. ]+", Path(file_name).stem) if s]
    useful = [s for s in segments if not _NEUTRAL_SEGMENTS.match(s)]
    return normalize_key(" ".join(useful)) if useful else None


def extract_subject_from_content(text: str) -> str | None:
    """Deduit la matiere des intitules, par vote de frequence.

    Un document mentionne son intitule reel plusieurs fois (page de garde,
    en-tetes, pieds de page), alors que les tournures parasites sont
    dispersees. Le candidat le plus frequent l'emporte, apres rejet des
    termes trop generiques.

    Args:
        text: Texte des premieres pages, sans accents

    Returns:
        La matiere normalisee, ou None
    """
    candidates = []
    for match in _TITLE_PATTERN.finditer(text):
        raw = " ".join(match.group(1).split())
        key = normalize_key(raw)
        if key and key not in _INVALID_SUBJECTS and 3 <= len(key) <= 45:
            candidates.append(key)
    if not candidates:
        return None
    best, occurrences = Counter(candidates).most_common(1)[0]
    logger.debug("Intitule retenu : '{}' ({} occurrences)", best, occurrences)
    return best


def identify_document(
    header_text: str,
    file_name: str,
    profile: DocumentProfile = DocumentProfile.MIXED,
) -> DocumentIdentity:
    """Deduit l'identite d'un document.

    Args:
        header_text: Texte des premieres pages du document
        file_name: Nom du fichier source
        profile: Profil structurel detecte a l'analyse

    Returns:
        L'identite du document
    """
    identity = DocumentIdentity(
        language=analyze_document_language(header_text), profile=profile
    )
    normalized = strip_accents(header_text)

    # -- Matiere : nom de fichier d'abord ------------------------------
    if subject := extract_subject_from_filename(file_name):
        identity.subject, identity.subject_source = subject, "filename"
    elif subject := extract_subject_from_content(normalized):
        identity.subject, identity.subject_source = subject, "content"

    # -- Niveau ---------------------------------------------------------
    if match := _LEVEL_PATTERN.search(normalized):
        identity.level = normalize_key(match.group(1)).rstrip("s")
    elif re.search(r"(?i)[_\-]Tles?[_\-.]|[_\-]Terminales?[_\-.]", file_name):
        identity.level = "terminale"

    # -- Serie ----------------------------------------------------------
    quoted = [m.group(1).upper() for m in _TRACK_QUOTED.finditer(normalized)]
    if quoted:
        identity.track = Counter(quoted).most_common(1)[0][0]
    elif match := _TRACK_ANNOUNCED.search(normalized):
        raw = " ".join(match.group(1).split()).upper()
        # Garde-fou : au-dela de 12 caracteres, la capture a deborde.
        identity.track = raw if len(raw) <= 12 else None
    if not identity.track and (match := _TRACK_FILENAME.search(file_name)):
        identity.track = match.group(1).upper()

    # -- Cycle ----------------------------------------------------------
    if match := _CYCLE_PATTERN.search(normalized):
        identity.cycle = normalize_key(match.group(1))
    elif identity.level in ("terminale", "premiere", "seconde"):
        identity.cycle = "secondaire"

    # -- Annee ----------------------------------------------------------
    years = [int(m.group(0)) for m in _YEAR_PATTERN.finditer(normalized)]
    if years:
        identity.program_year = Counter(years).most_common(1)[0][0]

    logger.info(
        "Identite de {} : {} (matiere depuis {}, langue {}, profil {})",
        file_name, identity.summary(), identity.subject_source,
        identity.language, profile.value,
    )
    return identity