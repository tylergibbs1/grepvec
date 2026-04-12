//! Implicit-mode cache storage backed by SQLite and embedding blob files.

use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::Connection;

use crate::error::StorageError;
use crate::types::Chunk;

use super::sqlite_store;
use super::{StorageBackend, StoredChunk};

/// Implicit-mode storage: SQLite index + embedding blob files under `~/.cache/grepvec/`.
pub struct ImplicitCache {
    root: PathBuf,
    conn: Connection,
}

impl ImplicitCache {
    /// Open (or create) the implicit cache at `cache_dir`.
    pub fn open(cache_dir: &Path) -> Result<Self, StorageError> {
        fs::create_dir_all(cache_dir)?;
        fs::create_dir_all(cache_dir.join("embeddings"))?;

        let db_path = cache_dir.join("index.db");
        let conn = Connection::open(&db_path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        sqlite_store::init_schema(&conn)?;

        Ok(Self {
            root: cache_dir.to_path_buf(),
            conn,
        })
    }

    /// Check that the stored model matches. If different, clear all data.
    pub fn ensure_model(&mut self, model_name: &str) -> Result<bool, StorageError> {
        let stored = sqlite_store::get_meta(&self.conn, "model");

        match stored.as_deref() {
            Some(s) if s == model_name => Ok(false),
            None => {
                sqlite_store::set_meta(&self.conn, "model", model_name)?;
                Ok(false)
            }
            Some(_) => {
                self.conn
                    .execute_batch("DELETE FROM chunks; DELETE FROM documents;")?;
                let emb_dir = self.embeddings_dir();
                if emb_dir.exists() {
                    fs::remove_dir_all(&emb_dir)?;
                    fs::create_dir_all(&emb_dir)?;
                }
                // Clear persistent vector index.
                let tv_path = self.root.join("vectors.tv");
                if tv_path.exists() {
                    fs::remove_file(&tv_path)?;
                }
                sqlite_store::set_meta(&self.conn, "model", model_name)?;
                sqlite_store::set_meta(&self.conn, "tv_chunk_count", "0")?;
                Ok(true)
            }
        }
    }

    fn embeddings_dir(&self) -> PathBuf {
        self.root.join("embeddings")
    }

    /// Expose the sqlite connection for direct queries.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Begin an explicit transaction for batching multiple writes.
    /// Call `commit_batch()` when done.
    pub fn begin_batch(&self) -> Result<(), StorageError> {
        self.conn.execute_batch("BEGIN")?;
        Ok(())
    }

    /// Commit a batch started with `begin_batch()`.
    pub fn commit_batch(&self) -> Result<(), StorageError> {
        self.conn.execute_batch("COMMIT")?;
        Ok(())
    }

    /// Path to the persistent turbovec index file.
    pub fn tv_index_path(&self) -> PathBuf {
        self.root.join("vectors.tv")
    }

    /// Check if the persistent .tv index is still valid (chunk count matches).
    pub fn is_tv_index_fresh(&self) -> bool {
        let stored = sqlite_store::get_meta(&self.conn, "tv_chunk_count")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let current = self.chunk_count().unwrap_or(0);
        stored > 0 && stored == current && self.tv_index_path().exists()
    }

    /// Mark the .tv index as built for the current chunk count.
    pub fn mark_tv_index_built(&self) -> Result<(), StorageError> {
        let count = self.chunk_count()?;
        sqlite_store::set_meta(&self.conn, "tv_chunk_count", &count.to_string())
    }

    /// Path to the query embedding cache directory.
    pub fn qcache_dir(&self) -> PathBuf {
        self.root.join("qcache")
    }
}

impl StorageBackend for ImplicitCache {
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn temp_cache() -> (tempfile::TempDir, ImplicitCache) {
        let dir = tempfile::tempdir().expect("test: failed to create temp dir");
        let cache = ImplicitCache::open(dir.path()).expect("test: failed to open cache");
        (dir, cache)
    }

    #[test]
    fn upsert_returns_true_for_new_document() {
        let (_dir, mut cache) = temp_cache();
        let chunks = vec![Chunk {
            doc_hash: "abc123".into(),
            chunk_idx: 0,
            text: "hello world".into(),
            byte_offset: 0,
        }];
        let embeddings = vec![vec![1.0f32; 4]];
        let changed = cache
            .upsert_document(Path::new("/tmp/test.txt"), "abc123", &chunks, &embeddings, 1000, 42)
            .expect("test: upsert failed");
        assert!(changed);
    }

    #[test]
    fn has_hash_returns_true_after_upsert() {
        let (_dir, mut cache) = temp_cache();
        let chunks = vec![Chunk {
            doc_hash: "abc123".into(),
            chunk_idx: 0,
            text: "hello world".into(),
            byte_offset: 0,
        }];
        let embeddings = vec![vec![1.0f32; 4]];
        cache
            .upsert_document(Path::new("/tmp/test.txt"), "abc123", &chunks, &embeddings, 0, 0)
            .expect("test: upsert failed");
        assert!(cache.has_hash("abc123").expect("test: has_hash failed"));
    }

    #[test]
    fn upsert_returns_false_for_unchanged_hash() {
        let (_dir, mut cache) = temp_cache();
        let chunks = vec![Chunk {
            doc_hash: "abc123".into(),
            chunk_idx: 0,
            text: "hello world".into(),
            byte_offset: 0,
        }];
        let embeddings = vec![vec![1.0f32; 4]];
        cache
            .upsert_document(Path::new("/tmp/test.txt"), "abc123", &chunks, &embeddings, 0, 0)
            .expect("test: upsert failed");
        let changed = cache
            .upsert_document(Path::new("/tmp/test.txt"), "abc123", &chunks, &embeddings, 0, 0)
            .expect("test: second upsert failed");
        assert!(!changed);
    }

    #[test]
    fn freshness_check_hits_on_matching_mtime_size() {
        let (_dir, mut cache) = temp_cache();
        let chunks = vec![Chunk {
            doc_hash: "h1".into(),
            chunk_idx: 0,
            text: "test".into(),
            byte_offset: 0,
        }];
        cache
            .upsert_document(Path::new("/tmp/a.txt"), "h1", &chunks, &[vec![1.0; 4]], 12345, 100)
            .expect("test: upsert");

        let hash = cache.check_freshness(Path::new("/tmp/a.txt"), 12345, 100);
        assert_eq!(hash.as_deref(), Some("h1"));
    }

    #[test]
    fn freshness_check_misses_on_changed_mtime() {
        let (_dir, mut cache) = temp_cache();
        let chunks = vec![Chunk {
            doc_hash: "h1".into(),
            chunk_idx: 0,
            text: "test".into(),
            byte_offset: 0,
        }];
        cache
            .upsert_document(Path::new("/tmp/a.txt"), "h1", &chunks, &[vec![1.0; 4]], 12345, 100)
            .expect("test: upsert");

        let hash = cache.check_freshness(Path::new("/tmp/a.txt"), 99999, 100);
        assert!(hash.is_none());
    }

    #[test]
    fn load_embedding_roundtrip() {
        let (_dir, mut cache) = temp_cache();
        let emb = vec![1.0f32, 2.0, 3.0, 4.0];
        let chunks = vec![Chunk {
            doc_hash: "h1".into(),
            chunk_idx: 0,
            text: "test".into(),
            byte_offset: 0,
        }];
        cache
            .upsert_document(Path::new("/tmp/a.txt"), "h1", &chunks, std::slice::from_ref(&emb), 0, 0)
            .expect("test: upsert failed");
        let loaded = cache.load_embedding("h1", 0, 4).expect("test: load failed");
        assert_eq!(loaded, emb);
    }

    #[test]
    fn all_chunks_returns_correct_count() {
        let (_dir, mut cache) = temp_cache();
        for i in 0..3 {
            let hash = format!("h{i}");
            let chunks = vec![Chunk {
                doc_hash: hash.clone(),
                chunk_idx: 0,
                text: format!("doc {i}"),
                byte_offset: 0,
            }];
            cache
                .upsert_document(
                    &PathBuf::from(format!("/tmp/{i}.txt")),
                    &hash,
                    &chunks,
                    &[vec![i as f32; 4]],
                    0, 0,
                )
                .expect("test: upsert failed");
        }
        let all = cache.all_chunks_ordered().expect("test: query failed");
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn all_chunks_are_ordered_by_row_id() {
        let (_dir, mut cache) = temp_cache();
        for i in 0..3 {
            let hash = format!("h{i}");
            let chunks = vec![Chunk {
                doc_hash: hash.clone(),
                chunk_idx: 0,
                text: format!("doc {i}"),
                byte_offset: 0,
            }];
            cache
                .upsert_document(
                    &PathBuf::from(format!("/tmp/{i}.txt")),
                    &hash,
                    &chunks,
                    &[vec![i as f32; 4]],
                    0, 0,
                )
                .expect("test: upsert failed");
        }
        let all = cache.all_chunks_ordered().expect("test: query failed");
        for w in all.windows(2) {
            assert!(w[0].row_id < w[1].row_id);
        }
    }

}
