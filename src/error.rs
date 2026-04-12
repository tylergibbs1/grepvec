//! Typed error hierarchy for the grepvec library.

use std::path::PathBuf;

/// Top-level error type for all grepvec library operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unknown model: {0}. valid: bge-large, mxbai-large, e5-large, gte-large, arctic-large, modernbert-large, bge-m3, bge-small")]
    UnknownModel(String),

    #[error("embedding failed: {0}")]
    Embedding(String),

    #[error("external embedder not yet implemented")]
    ExternalEmbedderUnimplemented,

    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("database error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("walk error: {0}")]
    Walk(#[from] ignore::Error),

    #[error("vector index error: {0}")]
    VecIndex(String),

    #[error("{0}")]
    Other(String),
}

/// Storage-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("corpus already exists at {0}")]
    CorpusAlreadyExists(PathBuf),

    #[error("no corpus found at {path} (missing index.db)")]
    CorpusNotFound { path: PathBuf },

    #[error("corpus has no model recorded")]
    NoModelRecorded,

    #[error("embedding blob too small for chunk_idx={chunk_idx} dim={dim}: {blob_bytes} bytes")]
    EmbeddingBlobTooSmall {
        chunk_idx: u32,
        dim: usize,
        blob_bytes: usize,
    },

    #[error("missing embedding blob: {0}")]
    MissingEmbeddingBlob(PathBuf),

    #[error("database error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("system time error: {0}")]
    Time(#[from] std::time::SystemTimeError),
}

pub type Result<T> = std::result::Result<T, Error>;
