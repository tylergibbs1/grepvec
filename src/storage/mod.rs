//! Storage abstraction layer for implicit cache and explicit corpus modes.

pub mod cache;
pub mod corpus;
pub(crate) mod sqlite_store;

use std::path::{Path, PathBuf};

use crate::error::StorageError;
use crate::types::Chunk;

/// A stored chunk with its database row ID (used to map to turbovec positions).
#[derive(Debug)]
pub struct StoredChunk {
    pub row_id: i64,
    pub doc_path: PathBuf,
    pub chunk_idx: u32,
    pub text: String,
    pub byte_offset: u64,
}

/// Abstraction over implicit cache and explicit corpus storage.
pub trait StorageBackend {
    /// Store a document's chunks and their embeddings.
    /// Returns true if this was a new/changed document (not a cache hit).
    fn upsert_document(
        &mut self,
        path: &Path,
        hash: &str,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
        mtime_secs: i64,
        file_size: i64,
    ) -> Result<bool, StorageError>;

    /// Check if a file is unchanged via mtime+size fast path. Returns stored hash if fresh.
    fn check_freshness(&self, path: &Path, mtime_secs: i64, file_size: i64) -> Option<String>;

    /// Check if a document with this hash is already stored.
    fn has_hash(&self, hash: &str) -> Result<bool, StorageError>;

    /// Load all stored chunks ordered by row ID (for building turbovec index).
    fn all_chunks_ordered(&self) -> Result<Vec<StoredChunk>, StorageError>;

    /// Load the raw f32 embedding for a chunk by its row ID.
    fn load_embedding(
        &self,
        doc_hash: &str,
        chunk_idx: u32,
        dim: usize,
    ) -> Result<Vec<f32>, StorageError>;

    /// Remove documents whose paths are not in `live_paths`.
    fn remove_stale(&mut self, live_paths: &[PathBuf]) -> Result<usize, StorageError>;

    /// Total number of chunks stored.
    fn chunk_count(&self) -> Result<usize, StorageError>;

    /// Get the storage root path.
    fn root(&self) -> &Path;
}
