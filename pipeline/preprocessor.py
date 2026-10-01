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
# Reparation d'encodage (mojibake)
# ======================================================================

# Certains PDF encodent leurs lettres accentuees avec une police Mac Roman
# lue ensuite comme du CP1252 : "é" devient "Ž", "•" devient "¥", etc. La
# substitution est SYSTEMATIQUE, donc reparable par table. Les SYMBOLES
# mathematiques (∫, ≤, ∑, α, →...), eux, sont correctement extraits : la
# table ne les touche pas.
_MOJIBAKE_MAP = {
    "Ž": "é", "¥": "•", "ˆ": "à", "ƒ": "É", "Õ": "'",
    "™": "ô", "”": "î", "‘": "ë",
}

# Caracteres dont la seule presence trahit un texte en mojibake : ils
# n'apparaissent jamais legitimement dans un texte francais. La reparation
# n'est appliquee QU'A un texte qui en contient : un document sain (avec ses
# vrais guillemets « », œ...) n'est donc jamais modifie.
_MOJIBAKE_MARKERS = ("Ž", "¥", "ƒ", "ˆ", "Õ")


def repair_encoding(text: str) -> str:
    """Repare un texte victime du mojibake Mac Roman -> CP1252.

    Ne fait rien si le texte ne porte aucune signature de mojibake : les
    documents sains passent inchanges. Sur un texte corrompu, chaque caractere
    parasite est remplace par la lettre accentuee qu'il represente.

    Args:
        text: Texte potentiellement corrompu

    Returns:
        Texte repare, ou le texte d'origine s'il est sain
    """
    if not text or not any(marker in text for marker in _MOJIBAKE_MARKERS):
        return text
    return "".join(_MOJIBAKE_MAP.get(char, char) for char in text)


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

# ======================================================================
# Vocabulaire CONTROLE des matieres + type de document
# ======================================================================
#
# L'ancienne extraction « par elimination » laissait le nom de fichier quasi
# brut dans `subject` (300+ valeurs : anglais_lv2_1er_gr, corrige_anglais...).
# On passe a une DETECTION par mots-cles vers une liste fermee de matieres :
# meme matiere -> meme valeur, quel que soit le libelle du fichier.
#
# Ordre = PRIORITE (du plus specifique au plus general) : on retient la
# premiere matiere dont un indice apparait (sur le slug, sinon le contenu).
_SUBJECT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sciences_physiques", ("sciences physiques", "sci phy", "sc phy", "sc phys",
                             "scien ph", "sci ph", "scph", "physique chimie", "physique")),
    ("svt", ("svt", "sciences de la vie", "science de la vie", "scvt")),
    ("mathematiques", ("mathematiques", "mathematique", "maths", "math")),
    ("philosophie", ("philosophie", "philo")),
    ("francais", ("francais", "fran", "lettres modernes", "lettres")),
    ("anglais", ("anglais", "english", "angl")),
    ("allemand", ("allemand", "allemang", "deutsch", "all lv")),
    ("espagnol", ("espagnol", "espanol", "espag", "esp")),
    ("portugais", ("portugais", "portugai", "port")),
    ("italien", ("italien", "italian")),
    ("russe", ("russe",)),
    ("grec", ("grec",)),
    ("latin", ("latin",)),
    ("arabe", ("arabe", "arab", "ara")),
    ("etudes_islamiques", ("etudes islamiques", "etude islamique", "islamique")),
    ("histoire_geographie", ("histoire geographie", "histoire", "geographie",
                             "hist geo", "hist-geo", "histo", "hg")),
    ("sciences_economiques_sociales", ("sciences economiques", "sciences eco",
                                       "economie generale", "economie", "ses")),
    ("informatique", ("informatique",)),
    ("droit", ("droit",)),
    ("gestion", ("management", "gestion")),
    ("education_civique", ("education civique", "civisme", "civilisation", "civilsation")),
    ("eps", ("education physique", "eps")),
    ("comptabilite", ("comptabilite", "compta")),
    ("construction_mecanique", ("construction mecanique", "cons meca", "cmc", "construction")),
)

# Jetons de STRUCTURE/TYPE a retirer pour deviner une matiere inconnue du
# vocabulaire (matieres techniques non listees), sans la polluer.
_STRUCTURE_TOKENS = {
    "programme", "programmes", "prog", "officiel", "final", "version", "senegal",
    "men", "document", "doc", "sujet", "sujets", "epreuve", "epreuves", "corrige",
    "corriges", "correction", "corr", "answer", "key", "bac", "baccalaureat",
    "1er", "2e", "2eme", "premier", "deuxieme", "groupe", "gr", "grp", "rempl",
    "remplacement", "session", "lv", "lv1", "lv2", "l1", "l2", "s1", "s2", "s3",
    "s4", "s5", "ls", "g", "steg", "stidd", "f6", "v", "r", "vl1", "canevas",
    "calendrier", "convocation", "diplomes", "vague", "region", "dakar",
    "terminale", "terminales", "tle", "tles", "premiere", "seconde", "classe",
    "classes", "de", "du", "des", "la", "le", "et", "a", "pour", "locale",
}


def _normalize_for_match(text: str) -> str:
    """Minuscule, sans accents, separateurs unifies en espaces."""
    norm = strip_accents(text or "").lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", norm).strip()


def canonical_subject(raw_slug: str | None) -> str | None:
    """Rabat un slug de nom de fichier sur une matiere canonique.

    Matching par TOKEN (et non par sous-chaine) pour eviter les faux positifs :
    un alias court comme « esp » ne doit matcher que le jeton « esp », jamais
    l'interieur d'un mot. A defaut, on nettoie le slug de ses jetons de
    structure/type pour isoler une matiere non listee.
    """
    phrase = _normalize_for_match(raw_slug)
    # Sous-chaine sur le SLUG uniquement : les noms de fichiers sont des libelles
    # courts et controles, donc « esp » / « ara » n'y matchent pas l'interieur
    # d'un mot (contrairement au contenu, d'ou son exclusion ici).
    for canon, keys in _SUBJECT_KEYWORDS:
        if any(key in phrase for key in keys):
            return canon
    # Matiere hors vocabulaire : on nettoie le slug plutot que de tout jeter.
    cleaned = "_".join(
        t for t in re.split(r"[^a-z0-9]+", strip_accents(raw_slug or "").lower())
        if t and t not in _STRUCTURE_TOKENS and not t.isdigit()
    )
    return cleaned if len(cleaned) >= 3 else None


def detect_document_type(file_name: str, text: str = "") -> str:
    """Devine le type de document : corrige / administratif / programme / sujet / cours.

    Ordre important : un corrige d'epreuve doit etre classe « corrige », pas
    « sujet » ; d'ou la verification des corriges en premier.
    """
    name = _normalize_for_match(file_name)
    if any(k in name for k in ("corrige", "corrig", "correction", "corrgige",
                               "corrife", "answer", "corr ")):
        return "corrige"
    if any(k in name for k in ("calendrier", "convocation", "diplome",
                               "deliberation", "releve", "vague", "proces", "pv ")):
        return "administratif"
    if any(k in name for k in ("programme", "prog ", "curriculum", "referentiel",
                               "canevas", "cadrage")):
        return "programme"
    if any(k in name for k in ("sujet", "epreuve", "bac", "groupe", " gr ", "gr ",
                               "1er", "2e", "rempl", "session", "examen",
                               "composition", "devoir")):
        return "sujet"
    if any(k in name for k in ("cours", "lecon", "fiche", "sequence")):
        return "cours"
    if "corrige" in _normalize_for_match(text[:2000]):
        return "corrige"
    return "inconnu"


# Niveaux du cycle elementaire : faux positifs frequents sur des copies du Bac.
_ELEMENTARY_LEVELS = {"ci", "cp", "ce1", "ce2", "cm1", "cm2"}


def _infer_level(level: str | None, document_type: str) -> str | None:
    """Fiabilise le niveau : les epreuves/corriges du Bac sont en terminale.

    Corrige les faux positifs « ci/cp/... » captes au fil du texte sur des
    sujets de Bac, et comble le niveau manquant pour ces documents.
    """
    if document_type in ("sujet", "corrige"):
        if level is None or level in _ELEMENTARY_LEVELS:
            return "terminale"
    return level


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
    document_type: str = "inconnu"  # programme | sujet | corrige | administratif | cours

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

    # -- Type de document (programme / sujet / corrige / ...) ----------
    identity.document_type = detect_document_type(file_name, header_text)

    # -- Matiere : vocabulaire controle sur le slug, sinon vote sur le contenu
    raw_slug = extract_subject_from_filename(file_name)
    if subject := canonical_subject(raw_slug):
        identity.subject, identity.subject_source = subject, "filename"
    elif title := extract_subject_from_content(normalized):
        # On canonicalise aussi l'intitule trouve dans le texte.
        identity.subject = canonical_subject(title) or title
        identity.subject_source = "content"

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

    # -- Fiabilisation du niveau (epreuves/corriges du Bac = terminale) -
    identity.level = _infer_level(identity.level, identity.document_type)

    logger.info(
        "Identite de {} : {} (matiere depuis {}, langue {}, profil {})",
        file_name, identity.summary(), identity.subject_source,
        identity.language, profile.value,
    )
    return identity