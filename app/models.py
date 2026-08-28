"""
Contrat de donnees pivot du pipeline JangAI.

Tous les etages du pipeline (extraction -> structuration -> chunking ->
embedding -> indexation) echangent des objets `Document(page_content,
metadata)`. La signature reproduit celle de langchain_core.documents.Document
afin qu'une brique LangChain (LangGraph, agents) puisse etre branchee sans
conversion.

Le contrat est volontairement GENERIQUE : aucune matiere fermee, aucun
schema de tableau predefini, aucune profondeur hierarchique fixe. Un
document de structure inconnue traverse le pipeline sans qu'aucune
enumeration n'ait a etre etendue.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


# ======================================================================
# Enumerations
# ======================================================================


class PageLayout(str, Enum):
    """Nature d'une page, mesuree et non declaree."""

    PROSE = "prose"
    TABLE = "table"
    MIXED = "mixed"
    VECTORIZED = "vectorized"   # texte converti en courbes -> OCR
    SCANNED = "scanned"         # image pleine page -> OCR
    EMPTY = "empty"             # aucun contenu -> a filtrer

    @property
    def needs_ocr(self) -> bool:
        return self in (PageLayout.VECTORIZED, PageLayout.SCANNED)

    @property
    def is_usable(self) -> bool:
        return self is not PageLayout.EMPTY


class BlockType(str, Enum):
    """Nature d'un bloc de texte, deduite de sa forme.

    C'est l'equivalent des balises semantiques absentes du format PDF :
    ce que serait H1/H2/H3, <li> ou <p> dans un document balise.
    """

    HEADING = "heading"           # titre, tous niveaux confondus
    PARAGRAPH = "paragraph"       # bloc de prose
    LIST_ITEM = "list_item"       # puce ou item numerote
    ARTICLE = "article"           # article de texte normatif
    ALINEA = "alinea"             # alinea a l'interieur d'un article
    TABLE_ROW = "table_row"       # ligne de grille
    CAPTION = "caption"           # legende, note de bas de page
    NOISE = "noise"               # pied de page, filet, numero isole


class DocumentProfile(str, Enum):
    """Profil structurel d'un document, detecte a l'analyse.

    Determine la strategie de decoupage. Le profil n'est jamais declare :
    il resulte du comptage des types de blocs rencontres.
    """

    TABULAR = "tabular"       # majorite de grilles (programmes scolaires)
    LEGAL = "legal"           # articles et alineas (lois, decrets)
    STRUCTURED = "structured" # titres hierarchiques et listes (rapports)
    NARRATIVE = "narrative"   # prose continue (discours, essais)
    MIXED = "mixed"           # combinaison sans dominante


class ChunkType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    ARTICLE = "article"
    TABLE_ROW = "table_row"
    TABLE_BLOCK = "table_block"


class ExtractionMethod(str, Enum):
    PYMUPDF_TEXT = "pymupdf_text"
    PYMUPDF_TABLE = "pymupdf_table"
    OCR_TESSERACT = "ocr_tesseract"
    DOCLING = "docling"           # structure reconstruite par Docling
    DOCLING_TABLE = "docling_table"


class Reliability(str, Enum):
    """Confiance accordee au contenu extrait.

    Alimente le filtrage et la ponderation du reranking : une ligne issue
    d'OCR merite moins de confiance qu'une extraction native.
    """

    HIGH = "high"       # extraction native, structure certaine
    MEDIUM = "medium"   # OCR reussi, ou ligne recollee entre deux pages
    LOW = "low"         # OCR douteux, structure incertaine


# ======================================================================
# Normalisation
# ======================================================================


def normalize_key(text: str) -> str:
    """Normalise un libelle en cle exploitable.

    Transforme un en-tete de colonne lu dans le document ("Competences
    exigibles") en cle de dictionnaire ("competences_exigibles"). C'est ce
    qui permet de nommer les colonnes sans les avoir declarees.

    Args:
        text: Libelle brut

    Returns:
        Cle normalisee, sans accent ni ponctuation
    """
    stripped = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )
    key = re.sub(r"[^a-z0-9]+", "_", stripped.lower()).strip("_")
    return key or "column"


# ======================================================================
# Position
# ======================================================================


@dataclass
class Position:
    """Localisation physique et logique d'un contenu.

    `page_number` est l'index reel dans le fichier ; `printed_page` est la
    pagination imprimee sur la page. Les deux divergent des qu'un fichier
    concatene plusieurs documents.
    """

    page_number: int
    printed_page: int | None = None
    printed_total: int | None = None
    sub_document_id: str | None = None
    row_index: int | None = None
    order: int | None = None


# ======================================================================
# Hierarchie
# ======================================================================


@dataclass
class HierarchyLevel:
    """Un niveau du chemin structurel, tel que decouvert dans le document.

    `depth` provient du classement des familles de numerotation observees.
    `number` est le marqueur brut ("VI", "3.3", "a)", "Article 12").
    """

    title: str
    depth: int
    number: str | None = None
    block_type: BlockType = BlockType.HEADING

    def label(self) -> str:
        return f"{self.number} {self.title}".strip() if self.number else self.title


@dataclass
class Hierarchy:
    """Chemin structurel complet, de profondeur variable."""

    levels: list[HierarchyLevel] = field(default_factory=list)

    def path(self) -> list[str]:
        """Libelles ordonnes, du plus general au plus specifique."""
        return [lvl.label() for lvl in sorted(self.levels, key=lambda x: x.depth)]

    def as_text(self, separator: str = " > ") -> str:
        return separator.join(self.path())

    def is_empty(self) -> bool:
        return not self.levels

    def descend(self, level: HierarchyLevel) -> Hierarchy:
        """Nouvelle hierarchie ou `level` remplace tout niveau de
        profondeur egale ou superieure.

        Un nouveau titre de niveau 1 invalide les niveaux 2 et suivants du
        contexte courant : c'est ce qui maintient un chemin coherent lors
        du parcours sequentiel du document.

        Args:
            level: Niveau a inserer

        Returns:
            Nouvelle hierarchie, l'originale restant inchangee
        """
        kept = [lvl for lvl in self.levels if lvl.depth < level.depth]
        return Hierarchy(levels=[*kept, level])


# ======================================================================
# Metadonnees
# ======================================================================


@dataclass
class ChunkMetadata:
    """Metadonnees portees par chaque chunk.

    Les champs de filtrage courant deviennent des colonnes typees et
    indexees dans PostgreSQL ; ce sont des chaines libres, remplies par
    extraction et non par une enumeration fermee. Le reste part en JSONB.
    """

    # --- Identite -----------------------------------------------------
    chunk_id: str
    document_id: str
    source_file: str

    # --- Classification, deduite du document ---------------------------
    subject: str | None = None       # "svt", "mathematiques", "physique"
    level: str | None = None         # "terminale", "premiere"
    track: str | None = None         # serie : "L2", "S2", "LS"
    cycle: str | None = None         # "secondaire", "elementaire"
    language: str | None = None      # "fr", "en"
    program_year: int | None = None
    document_profile: DocumentProfile = DocumentProfile.MIXED

    # --- Localisation -------------------------------------------------
    position: Position | None = None

    # --- Structure ----------------------------------------------------
    hierarchy: Hierarchy = field(default_factory=Hierarchy)
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    page_layout: PageLayout = PageLayout.PROSE

    # Les libelles de colonnes d'une ligne de tableau ne sont pas stockes
    # separement : ils sont integres au texte du chunk lors de
    # l'extraction ("Contenus : ...", "Activites : ..."). Ils restent donc
    # cherchables, sans champ ni index supplementaire.

    # --- Qualite ------------------------------------------------------
    extraction_method: ExtractionMethod = ExtractionMethod.PYMUPDF_TEXT
    reliability: Reliability = Reliability.HIGH
    merged_across_pages: bool = False
    char_count: int = 0
    content_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialisation plate, prete pour JSONB ou LangChain."""
        data = asdict(self)
        data["hierarchy_path"] = self.hierarchy.path()
        return data


# ======================================================================
# Document pivot
# ======================================================================


@dataclass
class Document:
    """Unite d'echange entre les etages du pipeline."""

    page_content: str
    metadata: ChunkMetadata

    def contextualized_text(self) -> str:
        """Texte reellement soumis au modele d'embedding.

        Une cellule de tableau isolee ("Regulation nerveuse") est
        semantiquement pauvre et se confond avec ses voisines. Prefixee de
        son chemin hierarchique, elle devient discriminante. `page_content`
        reste le contenu brut, restitue tel quel et cite en source.

        Returns:
            Texte prefixe du contexte documentaire et hierarchique
        """
        meta = self.metadata
        header: list[str] = []

        context = " ".join(
            x for x in (meta.subject, meta.level, meta.track) if x
        ).strip()
        if context:
            header.append(context)
        header.extend(meta.hierarchy.path())

        if not header:
            return self.page_content
        return " > ".join(header) + "\n\n" + self.page_content

    def to_langchain(self):
        """Conversion vers langchain_core.documents.Document.

        Import differe : LangChain n'est pas une dependance de l'ingestion.
        """
        from langchain_core.documents import Document as LCDocument

        return LCDocument(
            page_content=self.page_content, metadata=self.metadata.to_dict()
        )

    def to_dict(self) -> dict[str, Any]:
        return {"page_content": self.page_content, "metadata": self.metadata.to_dict()}

    def __len__(self) -> int:
        return len(self.page_content)

    def __repr__(self) -> str:
        meta = self.metadata
        page = meta.position.page_number if meta.position else "?"
        preview = " ".join(self.page_content.split())[:60]
        return f"Document({meta.subject or '?'} p.{page} {meta.chunk_type.value} : {preview!r})"


# ======================================================================
# Fabriques
# ======================================================================


def generate_document_id(file_path: str, sub_document: str | None = None) -> str:
    """Genere un identifiant stable pour un document ou sous-document.

    Args:
        file_path: Chemin ou nom du fichier source
        sub_document: Identifiant du sous-document, le cas echeant

    Returns:
        Empreinte hexadecimale de 16 caracteres
    """
    base = file_path if sub_document is None else f"{file_path}#{sub_document}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def generate_chunk_id(document_id: str, page_number: int, discriminator: str) -> str:
    """Genere un identifiant deterministe pour un chunk.

    Deterministe : reindexer deux fois ne cree aucun doublon.

    Args:
        document_id: Identifiant du document parent
        page_number: Numero de page PDF
        discriminator: Element distinguant les chunks d'une meme page

    Returns:
        Empreinte hexadecimale de 24 caracteres
    """
    seed = f"{document_id}:{page_number}:{discriminator}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def compute_content_hash(text: str) -> str:
    """Empreinte du contenu normalise, pour la deduplication.

    Args:
        text: Contenu du chunk

    Returns:
        Empreinte hexadecimale de 32 caracteres
    """
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def merge_documents(documents: Iterable[Document], separator: str = "\n\n") -> str:
    """Concatene le contenu de plusieurs documents (assemblage de contexte)."""
    return separator.join(d.page_content for d in documents)