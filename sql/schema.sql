-- ====================================================================
-- JangAI — Schema de la base vectorielle
--
-- Joue automatiquement a la premiere creation du conteneur.
-- Deux tables : `documents` (un fichier ou sous-document) et `chunks`
-- (les unites indexees).
--
-- Principe : les champs servant au FILTRAGE sont des colonnes typees et
-- indexees ; le reste part en JSONB. C'est ce qu'une abstraction de
-- vector store generique ne permet pas, et la raison d'avoir choisi
-- PostgreSQL plutot qu'une base purement vectorielle.
-- ====================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- recherche lexicale floue
CREATE EXTENSION IF NOT EXISTS unaccent;   -- indispensable en francais

-- Configuration plein texte francaise sans accents : permet a
-- "regulation" de retrouver "régulation".
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = 'fr_unaccent') THEN
    CREATE TEXT SEARCH CONFIGURATION fr_unaccent (COPY = french);
    ALTER TEXT SEARCH CONFIGURATION fr_unaccent
      ALTER MAPPING FOR hword, hword_part, word
      WITH unaccent, french_stem;
  END IF;
END
$$;

-- ====================================================================
-- Documents source
-- ====================================================================

CREATE TABLE IF NOT EXISTS documents (
    document_id       TEXT PRIMARY KEY,
    source_file       TEXT        NOT NULL,
    sub_document_id   TEXT,

    -- Classification extraite du document. Chaines libres : ajouter une
    -- matiere ou un type de document ne demande aucune migration.
    subject           TEXT,
    level             TEXT,
    track             TEXT,
    cycle             TEXT,
    language          TEXT,
    program_year      INTEGER,
    document_profile  TEXT,

    page_count        INTEGER     NOT NULL DEFAULT 0,
    chunk_count       INTEGER     NOT NULL DEFAULT 0,

    -- Pages internes absentes : un document incomplet doit etre connu
    -- comme tel, pour ne pas laisser croire a une couverture exhaustive.
    gaps              JSONB       NOT NULL DEFAULT '[]'::jsonb,
    diagnostics       JSONB       NOT NULL DEFAULT '{}'::jsonb,

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_subject ON documents (subject);
CREATE INDEX IF NOT EXISTS idx_documents_profile ON documents (document_profile);
CREATE INDEX IF NOT EXISTS idx_documents_source  ON documents (source_file);

-- ====================================================================
-- Chunks indexes
-- ====================================================================

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          TEXT PRIMARY KEY,
    document_id       TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    source_file       TEXT NOT NULL,

    -- `content`         : texte brut, restitue a l'utilisateur et cite.
    -- `indexed_content` : texte prefixe du chemin hierarchique, soumis au
    --                     modele d'embedding. Une cellule de tableau
    --                     isolee est semantiquement pauvre ; prefixee de
    --                     son chemin, elle devient discriminante.
    content           TEXT NOT NULL,
    indexed_content   TEXT NOT NULL,
    content_hash      TEXT,

    -- Colonnes de filtrage, typees et indexees.
    subject           TEXT,
    level             TEXT,
    track             TEXT,
    cycle             TEXT,
    language          TEXT,
    document_profile  TEXT,

    page_number       INTEGER,
    printed_page      INTEGER,
    sub_document_id   TEXT,
    chunk_order       INTEGER,

    chunk_type        TEXT NOT NULL DEFAULT 'paragraph',
    page_layout       TEXT,
    extraction_method TEXT,
    reliability       TEXT NOT NULL DEFAULT 'high',
    char_count        INTEGER NOT NULL DEFAULT 0,

    -- Structure, en JSONB.
    hierarchy         JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- 1024 dimensions : Qwen3-Embedding-0.6B. Reste sous la limite de
    -- 2000 dimensions imposee par pgvector pour l'index HNSW.
    embedding         vector(1024),

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- --------------------------------------------------------------------
-- Index de filtrage
-- --------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_chunks_subject     ON chunks (subject);
CREATE INDEX IF NOT EXISTS idx_chunks_track       ON chunks (track);
CREATE INDEX IF NOT EXISTS idx_chunks_language    ON chunks (language);
CREATE INDEX IF NOT EXISTS idx_chunks_document    ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type        ON chunks (chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_reliability ON chunks (reliability);
CREATE INDEX IF NOT EXISTS idx_chunks_hash        ON chunks (content_hash);

-- Filtre combine le plus courant.
CREATE INDEX IF NOT EXISTS idx_chunks_subject_track ON chunks (subject, track);

-- Recherche a l'interieur du chemin hierarchique.
--
-- Aucun index sur le detail des colonnes de tableau : les libelles de
-- colonnes sont conserves DANS le texte du chunk ("Contenus : ...",
-- "Competences exigibles : ..."), ce qui les rend cherchables par le
-- vectoriel et le plein texte. Les stocker en plus comme champ filtrable
-- n'aurait servi aucune requete du produit, et coutait un index GIN.
CREATE INDEX IF NOT EXISTS idx_chunks_hierarchy ON chunks USING GIN (hierarchy);

-- --------------------------------------------------------------------
-- Recherche lexicale, complement du vectoriel.
-- Le vectoriel echoue sur les termes rares et les codes ("RTI", "3.3") ;
-- le lexical les retrouve exactement. Les deux se combinent au reranking.
-- --------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_chunks_trgm
    ON chunks USING GIN (content gin_trgm_ops);
-- Plein-texte sur `indexed_content` : titres et contexte hierarchique
-- compris, la ou vivent souvent les termes les plus discriminants.
CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON chunks USING GIN (to_tsvector('fr_unaccent', indexed_content));

-- --------------------------------------------------------------------
-- Index vectoriel HNSW.
-- Cree ici a vide, ce qui est acceptable a cette echelle. Sur un volume
-- important, mieux vaut inserer puis construire : `rebuild_index()` le
-- fait, la construction en bloc etant plus rapide et de meilleure qualite.
-- --------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- --------------------------------------------------------------------
-- Vue de controle qualite : a consulter apres chaque ingestion.
-- --------------------------------------------------------------------
CREATE OR REPLACE VIEW v_ingestion_quality AS
SELECT
    d.source_file,
    d.subject,
    d.track,
    d.document_profile,
    d.page_count,
    COUNT(c.chunk_id)                                      AS chunk_count,
    COUNT(c.embedding)                                     AS embedded_count,
    ROUND(AVG(c.char_count))                               AS avg_chunk_size,
    MAX(c.char_count)                                      AS max_chunk_size,
    COUNT(*) FILTER (WHERE c.reliability <> 'high')        AS low_reliability_count,
    COUNT(*) FILTER (WHERE c.chunk_type = 'table_row')     AS table_row_count,
    jsonb_array_length(d.gaps)                             AS gap_count
FROM documents d
LEFT JOIN chunks c ON c.document_id = d.document_id
GROUP BY d.document_id, d.source_file, d.subject, d.track,
         d.document_profile, d.page_count, d.gaps
ORDER BY d.source_file;