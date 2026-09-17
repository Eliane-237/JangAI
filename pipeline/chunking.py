"""
Decoupage — mise en chunks du document structure.

Responsabilite unique : transformer un `DoclingDocument` (produit par
`pipeline.extractors.document_extractor`) en `Document` prets a vectoriser.

Le decoupage est confie au HybridChunker de Docling : a la fois
structure-aware (il respecte les frontieres naturelles — sections, listes,
tableaux) et token-aware (il borne chaque chunk selon le tokenizer du modele
d'embedding). Aucune heuristique de taille maison : on aligne simplement le
tokenizer sur le modele d'embedding pour qu'aucun chunk ne deborde sa fenetre.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from loguru import logger

from app.config import get_settings
from app.models import (
    ChunkMetadata,
    ChunkType,
    Document,
    DocumentProfile,
    ExtractionMethod,
    Hierarchy,
    HierarchyLevel,
    Position,
    Reliability,
    compute_content_hash,
    generate_chunk_id,
)
from pipeline.preprocessor import repair_encoding

# Labels Docling designant une table et une liste.
_TABLE_LABEL = "table"
_LIST_LABEL = "list_item"


# ======================================================================
# Chunker (charge une seule fois par processus)
# ======================================================================


@lru_cache(maxsize=1)
def get_chunker():
    """HybridChunker cale sur le tokenizer du modele d'embedding.

    Returns:
        Le `HybridChunker` configure
    """
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    settings = get_settings()
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        settings.embedding_model, max_tokens=settings.chunk_max_tokens
    )
    logger.info(
        "HybridChunker : tokenizer {} ({} tokens max)",
        settings.embedding_model, settings.chunk_max_tokens,
    )
    return HybridChunker(tokenizer=tokenizer, merge_peers=settings.chunk_merge_peers)


# ======================================================================
# Lecture des metadonnees d'un chunk
# ======================================================================


def _label(item) -> str:
    """Libelle normalise d'un element Docling."""
    label = getattr(item, "label", None)
    return str(getattr(label, "value", label) or "").lower()


def _chunk_signals(chunk) -> tuple[int | None, ChunkType]:
    """Page de depart et type d'un chunk, deduits de ses elements Docling.

    Args:
        chunk: Chunk produit par le HybridChunker

    Returns:
        Tuple (premiere page, type de chunk)
    """
    pages: list[int] = []
    labels: set[str] = set()
    for item in getattr(chunk.meta, "doc_items", []) or []:
        labels.add(_label(item))
        provenance = getattr(item, "prov", None) or []
        pages.extend(p.page_no for p in provenance if getattr(p, "page_no", None))

    if _TABLE_LABEL in labels:
        chunk_type = ChunkType.TABLE_BLOCK
    elif labels and labels <= {_LIST_LABEL}:
        chunk_type = ChunkType.LIST
    else:
        chunk_type = ChunkType.PARAGRAPH

    return (min(pages) if pages else None), chunk_type


def _hierarchy(chunk) -> Hierarchy:
    """Chemin hierarchique d'un chunk, tire des titres fournis par Docling."""
    headings = getattr(chunk.meta, "headings", None) or []
    return Hierarchy(
        levels=[
            HierarchyLevel(title=repair_encoding(title), depth=depth)
            for depth, title in enumerate(headings)
        ]
    )


# ======================================================================
# Profil du document
# ======================================================================


def infer_profile(documents: list[Document]) -> DocumentProfile:
    """Deduit le profil structurel de la nature des chunks produits."""
    counts = Counter(d.metadata.chunk_type for d in documents)
    total = sum(counts.values()) or 1
    table_share = (counts[ChunkType.TABLE_BLOCK] + counts[ChunkType.TABLE_ROW]) / total
    structured_share = (counts[ChunkType.LIST] + counts[ChunkType.HEADING]) / total

    if table_share >= 0.40:
        return DocumentProfile.TABULAR
    if structured_share >= 0.40:
        return DocumentProfile.STRUCTURED
    return DocumentProfile.NARRATIVE


# ======================================================================
# Point d'entree
# ======================================================================


def chunk_document(
    doc,
    document_id: str,
    source_file: str,
    identity,
    ocr_pages: set[int] | None = None,
) -> list[Document]:
    """Decoupe un `DoclingDocument` en `Document` prets a vectoriser.

    Args:
        doc: DoclingDocument converti et complete
        document_id: Identifiant du document parent
        source_file: Nom du fichier source
        identity: Identite deduite (`DocumentIdentity`), dont le profil est
            renseigne ici a partir des chunks reellement produits
        ocr_pages: Pages issues de l'OCR ; leurs chunks voient leur fiabilite
            abaissee, une reconnaissance ne valant jamais une extraction native

    Returns:
        La liste des chunks
    """
    settings = get_settings()
    chunker = get_chunker()
    ocr_pages = ocr_pages or set()

    documents: list[Document] = []
    order = 0

    for chunk in chunker.chunk(doc):
        # Reparation du mojibake (PDF a police Mac Roman) : sans effet sur les
        # documents sains, elle corrige les accents des PDF concernes (Maths).
        content = repair_encoding(chunk.text or "").strip()
        if len(content) < settings.chunk_min_size:
            continue

        page, chunk_type = _chunk_signals(chunk)
        is_table = chunk_type is ChunkType.TABLE_BLOCK
        from_ocr = page in ocr_pages

        metadata = ChunkMetadata(
            chunk_id=generate_chunk_id(document_id, page or 0, str(order)),
            document_id=document_id,
            source_file=source_file,
            subject=identity.subject,
            level=identity.level,
            track=identity.track,
            cycle=identity.cycle,
            language=identity.language,
            program_year=identity.program_year,
            position=Position(page_number=page or 0, order=order),
            hierarchy=_hierarchy(chunk),
            chunk_type=chunk_type,
            extraction_method=(
                ExtractionMethod.OCR_TESSERACT
                if from_ocr
                else ExtractionMethod.DOCLING_TABLE
                if is_table
                else ExtractionMethod.DOCLING
            ),
            reliability=Reliability.MEDIUM if from_ocr else Reliability.HIGH,
            char_count=len(content),
            content_hash=compute_content_hash(content),
        )
        documents.append(Document(page_content=content, metadata=metadata))
        order += 1

    # Le profil se lit sur ce que le document a reellement produit, puis se
    # propage a l'identite et a chaque chunk.
    profile = infer_profile(documents)
    identity.profile = profile
    for document in documents:
        document.metadata.document_profile = profile

    sizes = [len(d) for d in documents] or [0]
    logger.info(
        "{} : {} chunks | profil {} | moyenne {} car. (min {}, max {})",
        source_file, len(documents), profile.value,
        sum(sizes) // len(sizes), min(sizes), max(sizes),
    )
    return documents
