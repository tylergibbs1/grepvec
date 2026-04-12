//! Named corpus storage for explicit-mode operation.

use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::Connection;

use crate::error::StorageError;
use crate::types::Chunk;

use super::sqlite_store;
use super::{StorageBackend, StoredChunk};

/// Explicit corpus: same schema as `ImplicitCache`, but in a user-chosen directory.
pub struct NamedCorpus {
    root: PathBuf,
    conn: Connection,
}

/// Summary statistics for a named corpus.
#[derive(Debug)]
pub struct CorpusStats {
    pub doc_count: usize,
    pub chunk_count: usize,
    pub disk_bytes: u64,
}

impl NamedCorpus {
    /// Create a new corpus at `path`. Fails if it already exists.
    pub fn create(path: &Path, model_name: &str) -> Result<Self, StorageError> {
        if path.join("index.db").exists() {
            return Err(StorageError::CorpusAlreadyExists(path.to_path_buf()));
        }
        fs::create_dir_all(path)?;
        fs::create_dir_all(path.join("embeddings"))?;

        let conn = Connection::open(path.join("index.db"))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        sqlite_store::init_schema(&conn)?;

        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('model', ?1)",
            [model_name],
        )?;

        Ok(Self {
            root: path.to_path_buf(),
            conn,
        })
    }

    /// Open an existing corpus.
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        if !path.join("index.db").exists() {
            return Err(StorageError::CorpusNotFound {
                path: path.to_path_buf(),
            });
        }

        let conn = Connection::open(path.join("index.db"))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;

        Ok(Self {
            root: path.to_path_buf(),
            conn,
        })
    }

    /// Returns the model name stored in this corpus.
    pub fn model_name(&self) -> Result<String, StorageError> {
        self.conn
            .query_row("SELECT value FROM meta WHERE key = 'model'", [], |row| {
                row.get(0)
            })
            .map_err(|_| StorageError::NoModelRecorded)
    }

    /// Expose the sqlite connection for BM25 queries.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    fn embeddings_dir(&self) -> PathBuf {
        self.root.join("embeddings")
    }

    /// Compact: vacuum the database to reclaim space.
    pub fn compact(&self) -> Result<(), StorageError> {
        self.conn.execute_batch("VACUUM;")?;
        Ok(())
    }

    /// Stats about this corpus.
    pub fn stats(&self) -> Result<CorpusStats, StorageError> {
        let doc_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;
        let chunk_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;

        let mut disk_size: u64 = 0;
        if self.root.exists() {
            for entry in fs::read_dir(&self.root)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    disk_size += entry.metadata()?.len();
                }
            }
            let emb_dir = self.embeddings_dir();
            if emb_dir.exists() {
                for entry in fs::read_dir(&emb_dir)? {
                    let entry = entry?;
                    disk_size += entry.metadata()?.len();
                }
            }
        }

        Ok(CorpusStats {
            doc_count: doc_count as usize,
            chunk_count: chunk_count as usize,
            disk_bytes: disk_size,
        })
    }
}

impl StorageBackend for NamedCorpus {
    fn upsert_document(
        &mut self,
        path: &Path,
        hash: &str,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
        mtime_secs: i64,
        file_size: i64,
    ) -> Result<bool, StorageError> {
        sqlite_store::upsert_document(
            &self.conn,
            &self.embeddings_dir(),
            path,
            hash,
            chunks,
            embeddings,
            mtime_secs,
            file_size,
        )
    }

    fn check_freshness(&self, path: &Path, mtime_secs: i64, file_size: i64) -> Option<String> {
        sqlite_store::check_freshness(&self.conn, path, mtime_secs, file_size)
    }

    fn has_hash(&self, hash: &str) -> Result<bool, StorageError> {
        sqlite_store::has_hash(&self.conn, hash)
    }

    fn all_chunks_ordered(&self) -> Result<Vec<StoredChunk>, StorageError> {
        sqlite_store::all_chunks_ordered(&self.conn)
    }

    fn load_embedding(
        &self,
        doc_hash: &str,
        chunk_idx: u32,
        dim: usize,
    ) -> Result<Vec<f32>, StorageError> {
        sqlite_store::load_embedding(&self.embeddings_dir(), doc_hash, chunk_idx, dim)
    }

    fn remove_stale(&mut self, live_paths: &[PathBuf]) -> Result<usize, StorageError> {
        sqlite_store::remove_stale(&self.conn, live_paths)
    }

    fn chunk_count(&self) -> Result<usize, StorageError> {
        sqlite_store::chunk_count(&self.conn)
    }

    fn root(&self) -> &Path {
        &self.root
    }
}
