"""
Configuration centralisee du pipeline JangAI.

Point unique de verite pour les chemins, les modeles et les parametres
d'indexation.

Principe directeur : cette configuration ne connait aucune matiere, aucun
nom de fichier, aucun schema de tableau. Elle ne contient que des seuils
mesures et des parametres reglables. Toute la structure d'un document est
lue DANS le document, a l'execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ======================================================================
# Arborescence
# ======================================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DIAGNOSTICS_DIR = DATA_DIR / "diagnostics"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
OCR_CACHE_DIR = DATA_DIR / "ocr_cache"
LOGS_DIR = ROOT / "logs"

for _dir in (DIAGNOSTICS_DIR, INTERMEDIATE_DIR, OCR_CACHE_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Seuils de detection
# ======================================================================


@dataclass(frozen=True)
class DetectionThresholds:
    """Seuils gouvernant l'analyse d'une page et d'un bloc.

    Chaque valeur est calibree sur des mesures reelles, non choisie a
    l'intuition. Les plages observees sont indiquees pour que le reglage
    reste verifiable quand le corpus evoluera.
    """

    # --- Presence d'une couche texte exploitable ----------------------
    min_text_chars: int = 60
    # Densite = caracteres / surface de page. Ne rattrape que les pages ou
    # l'extracteur renvoie quelques fragments isoles sur une grande
    # surface. Calibre a ~60 caracteres sur une A4 (501 000 pt2) : un
    # seuil plus eleve ecarterait des pages de prose legitimes, dont les
    # plus courtes observees comptent 560 caracteres.
    min_char_density: float = 0.00012

    # --- Texte converti en courbes ------------------------------------
    # Une page sans texte ni image mais riche en traces vectoriels
    # contient du texte vectorise : invisible aux extracteurs, parfaitement
    # lisible a l'ecran. Seul l'OCR peut le recuperer.
    # Observe : 0-46 traces sur les pages natives, 257-1544 sur les pages
    # vectorisees. Seuil pose dans le creux separant les deux populations.
    min_vector_paths: int = 120
    max_paths_empty_page: int = 5

    # --- Page scannee --------------------------------------------------
    min_image_ratio_scanned: float = 0.50

    # --- Tableaux ------------------------------------------------------
    min_table_surface_ratio: float = 0.35
    # Bandes haute et basse, en fraction de la hauteur de page. Un tableau
    # qui les atteint signale une ligne coupee entre deux pages.
    top_band: float = 0.12
    bottom_band: float = 0.88
    min_table_columns: int = 2
    min_table_rows: int = 2

    # --- Blocs ---------------------------------------------------------
    max_heading_chars: int = 180
    max_hierarchy_depth: int = 5
    # Part de majuscules au-dela de laquelle une ligne courte est tenue
    # pour un titre, faute de numerotation.
    min_uppercase_ratio_heading: float = 0.75

    # --- Profil de document --------------------------------------------
    # Part minimale d'un type de bloc pour imposer le profil du document.
    min_profile_dominance: float = 0.30

    # --- Sous-documents ------------------------------------------------
    # Un groupe representant moins que cette part du fichier est tenu pour
    # un ensemble de pages parasites.
    min_sub_document_share: float = 0.15

    # --- Deduplication --------------------------------------------------
    # Similarite au-dela de laquelle deux chunks sont tenus pour doublons.
    min_duplicate_similarity: int = 92

    # --- OCR -------------------------------------------------------------
    min_ocr_confidence: float = 0.60


THRESHOLDS = DetectionThresholds()


# ======================================================================
# Parametres reglables via .env
# ======================================================================


class Settings(BaseSettings):
    """Parametres d'execution, surchargeables par variables d'environnement."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Embedding ----------------------------------------------------
    embedding_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    embedding_dim: int = Field(default=1024)
    embedding_batch_size: int = Field(default=16)
    embedding_device: str = Field(default="cpu")
    embedding_normalize: bool = Field(default=True)

    # Qwen3 attend une instruction cote REQUETE uniquement ; les documents
    # sont encodes bruts. L'appliquer des deux cotes degrade le rappel.
    query_instruction: str = Field(
        default=(
            "Instruct: Retrouve les extraits de programme scolaire "
            "repondant a la question\nQuery: "
        )
    )

    # --- Recherche et reranking ---------------------------------------
    search_top_k: int = Field(default=30)
    reranker_model: str = Field(default="Qwen/Qwen3-Reranker-0.6B")
    rerank_top_k: int = Field(default=8)

    # Ponderations du reranking hybride, sur des signaux generiques.
    weight_vector_score: float = Field(default=0.55)
    weight_hierarchy_match: float = Field(default=0.20)
    weight_term_density: float = Field(default=0.15)
    weight_reliability: float = Field(default=0.10)

    # --- PostgreSQL ---------------------------------------------------
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="jangai")
    postgres_user: str = Field(default="jangai")
    postgres_password: str = Field(default="change_me")

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # --- Index HNSW ---------------------------------------------------
    # m               : liens par noeud, 16 est le compromis standard.
    # ef_construction : largeur de recherche a la construction.
    # ef_search       : largeur a l'interrogation, reglable a chaud.
    hnsw_m: int = Field(default=16)
    hnsw_ef_construction: int = Field(default=64)
    hnsw_ef_search: int = Field(default=100)
    hnsw_operator: str = Field(default="vector_cosine_ops")

    # --- Chunking (HybridChunker Docling) -----------------------------
    # Le decoupage est structure-aware ET token-aware : Docling respecte les
    # frontieres naturelles du document (sections, listes, tableaux) tout en
    # bornant chaque chunk a `chunk_max_tokens` selon le tokenizer du modele
    # d'embedding. Aligner ce tokenizer sur Qwen3 evite qu'un chunk deborde
  
    chunk_max_tokens: int = Field(default=512)
    chunk_merge_peers: bool = Field(default=True)
    chunk_min_size: int = Field(default=80)   # plancher en caracteres

    # --- Extraction des tableaux --------------------------------------
    # Reconnaissance de structure par TableFormer (Docling). Resout les
    # cellules fusionnees et marque les en-tetes.
    docling_table_structure: bool = Field(default=True)

    # --- OCR ----------------------------------------------------------
    tesseract_cmd: str | None = Field(default=None)
    poppler_path: str | None = Field(default=None)
    ocr_dpi: int = Field(default=300)
    # psm 1 : segmentation automatique avec detection d'orientation.
    # psm 6 : bloc uniforme, nettement meilleur sur les grilles.
    ocr_psm_prose: int = Field(default=1)
    ocr_psm_table: int = Field(default=6)
    ocr_oem: int = Field(default=1)
    ocr_languages: str = Field(default="fra+eng")
    ocr_cache_enabled: bool = Field(default=True)

    # --- LLM (Groq) ---------------------------------------------------
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_temperature: float = Field(default=0.2)
    groq_max_tokens: int = Field(default=1500)
    max_context_chars: int = Field(default=8000)

    # --- Divers -------------------------------------------------------
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instance unique des parametres, mise en cache."""
    return Settings()


def check_configuration() -> list[str]:
    """Controles de coherence au demarrage.

    Returns:
        Liste des avertissements, vide si la configuration est saine
    """
    warnings: list[str] = []
    s = get_settings()

    # pgvector limite l'index HNSW a 2000 dimensions.
    if s.embedding_dim > 2000:
        warnings.append(
            f"embedding_dim={s.embedding_dim} depasse la limite HNSW de pgvector "
            f"(2000). Utilisez la troncature MRL ou un index IVFFlat."
        )
    if s.hnsw_ef_search < s.rerank_top_k:
        warnings.append(
            f"hnsw_ef_search={s.hnsw_ef_search} < rerank_top_k={s.rerank_top_k} : "
            f"le rappel sera bride."
        )
    if s.search_top_k < s.rerank_top_k:
        warnings.append("search_top_k doit etre superieur a rerank_top_k.")

    total = (
        s.weight_vector_score
        + s.weight_hierarchy_match
        + s.weight_term_density
        + s.weight_reliability
    )
    if abs(total - 1.0) > 0.01:
        warnings.append(f"Les poids de reranking totalisent {total:.2f} au lieu de 1.00.")

    if s.chunk_max_tokens < 64:
        warnings.append("chunk_max_tokens tres bas : les chunks risquent d'etre tronques.")
    if not s.groq_api_key:
        warnings.append("GROQ_API_KEY absente : la generation sera indisponible.")
    if not s.tesseract_cmd:
        warnings.append(
            "TESSERACT_CMD absent : l'OCR des pages scannees sera indisponible."
        )
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.pdf")):
        warnings.append(f"Aucun PDF trouve dans {RAW_DIR}")

    return warnings