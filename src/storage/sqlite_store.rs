//! Shared SQLite + blob storage logic used by both `ImplicitCache` and `NamedCorpus`.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::Connection;

use crate::error::StorageError;
use crate::types::Chunk;

use super::StoredChunk;

/// Initialize the shared database schema on the given connection.
pub fn init_schema(conn: &Connection) -> Result<(), StorageError> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY,
            path        TEXT NOT NULL,
            hash        TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            indexed_at  INTEGER NOT NULL,
            mtime_secs  INTEGER NOT NULL DEFAULT 0,
            file_size   INTEGER NOT NULL DEFAULT 0
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_path ON documents(path);
        CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(hash);

        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY,
            doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_idx   INTEGER NOT NULL,
            text        TEXT NOT NULL,
            byte_offset INTEGER NOT NULL,
            UNIQUE(doc_id, chunk_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(doc_id);

        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '3');

        PRAGMA foreign_keys = ON;
        ",
    )?;
    // Migration: add mtime/size columns if missing (upgrade from older schemas).
    let _ = conn.execute(
        "ALTER TABLE documents ADD COLUMN mtime_secs INTEGER NOT NULL DEFAULT 0",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE documents ADD COLUMN file_size INTEGER NOT NULL DEFAULT 0",
        [],
    );
    // Drop BM25 table if leftover from older schema.
    let _ = conn.execute("DROP TABLE IF EXISTS bm25_terms", []);
    Ok(())
}

/// Check if a file is unchanged based on mtime+size. Returns the stored hash if fresh.
pub fn check_freshness(
    conn: &Connection,
    path: &Path,
    mtime_secs: i64,
    file_size: i64,
) -> Option<String> {
    let path_str = path.to_string_lossy();
    let result: std::result::Result<(String, i64, i64), _> = conn.query_row(
        "SELECT hash, mtime_secs, file_size FROM documents WHERE path = ?1",
        [&*path_str],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    );
    match result {
        Ok((hash, stored_mtime, stored_size))
            if stored_mtime == mtime_secs && stored_size == file_size =>
        {
            Some(hash)
        }
        _ => None,
    }
}

/// Upsert a document's chunks and embeddings into the database + blob files.
pub fn upsert_document(
    conn: &Connection,
    emb_dir: &Path,
    path: &Path,
    hash: &str,
    chunks: &[Chunk],
    embeddings: &[Vec<f32>],
    mtime_secs: i64,
    file_size: i64,
) -> Result<bool, StorageError> {
    let path_str = path.to_string_lossy();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs() as i64;

    let existing_hash: Option<String> = conn
        .query_row(
            "SELECT hash FROM documents WHERE path = ?1",
            [&*path_str],
            |row| row.get(0),
        )
        .ok();

    if existing_hash.as_deref() == Some(hash) {
        conn.execute(
            "UPDATE documents SET mtime_secs = ?1, file_size = ?2 WHERE path = ?3",
            rusqlite::params![mtime_secs, file_size, &*path_str],
        )?;
        return Ok(false);
    }

    // If already inside a batch transaction, just execute directly.
    // Otherwise wrap in our own transaction.
    let in_transaction = conn.is_autocommit(); // false means we're in a transaction
    if in_transaction {
        // Not in a transaction yet — start one.
        conn.execute_batch("BEGIN")?;
    }

    if existing_hash.is_some() {
        conn.execute("DELETE FROM documents WHERE path = ?1", [&*path_str])?;
    }

    conn.execute(
        "INSERT INTO documents (path, hash, chunk_count, indexed_at, mtime_secs, file_size) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![&*path_str, hash, chunks.len() as i64, now, mtime_secs, file_size],
    )?;
    let doc_id = conn.last_insert_rowid();

    for chunk in chunks {
        conn.execute(
            "INSERT INTO chunks (doc_id, chunk_idx, text, byte_offset) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![doc_id, chunk.chunk_idx, chunk.text, chunk.byte_offset as i64],
        )?;
    }

    if in_transaction {
        conn.execute_batch("COMMIT")?;
    }

    let emb_path = emb_dir.join(format!("{hash}.emb"));
    let mut blob = Vec::with_capacity(embeddings.iter().map(|e| e.len() * 4).sum());
    for emb in embeddings {
        for &val in emb {
            blob.extend_from_slice(&val.to_le_bytes());
        }
    }
    fs::write(&emb_path, &blob)?;

    Ok(true)
}

/// Check if a document with this content hash is already stored.
pub fn has_hash(conn: &Connection, hash: &str) -> Result<bool, StorageError> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM documents WHERE hash = ?1",
        [hash],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

/// Load all stored chunks ordered by row ID.
pub fn all_chunks_ordered(conn: &Connection) -> Result<Vec<StoredChunk>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT c.id, d.path, c.chunk_idx, c.text, c.byte_offset
         FROM chunks c
         JOIN documents d ON c.doc_id = d.id
         ORDER BY c.id ASC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(StoredChunk {
            row_id: row.get(0)?,
            doc_path: PathBuf::from(row.get::<_, String>(1)?),
            chunk_idx: row.get(2)?,
            text: row.get(3)?,
            byte_offset: row.get::<_, i64>(4)? as u64,
        })
    })?;
    Ok(rows.collect::<Result<Vec<_>, _>>()?)
}

/// Load the raw f32 embedding for a chunk from its blob file.
pub fn load_embedding(
    emb_dir: &Path,
    doc_hash: &str,
    chunk_idx: u32,
    dim: usize,
) -> Result<Vec<f32>, StorageError> {
    let emb_path = emb_dir.join(format!("{doc_hash}.emb"));
    let bytes = fs::read(&emb_path)
        .map_err(|_| StorageError::MissingEmbeddingBlob(emb_path.clone()))?;

    let offset = chunk_idx as usize * dim * 4;
    let end = offset + dim * 4;
    if end > bytes.len() {
        return Err(StorageError::EmbeddingBlobTooSmall {
            chunk_idx,
            dim,
            blob_bytes: bytes.len(),
        });
    }

    Ok(bytes[offset..end]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Remove documents whose paths are not in `live_paths`.
pub fn remove_stale(
    conn: &Connection,
    live_paths: &[PathBuf],
) -> Result<usize, StorageError> {
    let all_docs: Vec<(i64, String)> = {
        let mut stmt = conn.prepare("SELECT id, path FROM documents")?;
        stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?
    };

    let live_set: HashSet<&Path> = live_paths.iter().map(PathBuf::as_path).collect();

    let mut removed = 0;
    for (doc_id, path_str) in &all_docs {
        if !live_set.contains(Path::new(path_str.as_str())) {
            conn.execute("DELETE FROM documents WHERE id = ?1", [doc_id])?;
            removed += 1;
        }
    }
    Ok(removed)
}

/// Total number of chunks stored.
pub fn chunk_count(conn: &Connection) -> Result<usize, StorageError> {
    let n: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
    Ok(n as usize)
}

/// Get a meta value.
pub fn get_meta(conn: &Connection, key: &str) -> Option<String> {
    conn.query_row(
        "SELECT value FROM meta WHERE key = ?1",
        [key],
        |row| row.get(0),
    )
    .ok()
}

/// Set a meta value.
pub fn set_meta(conn: &Connection, key: &str, value: &str) -> Result<(), StorageError> {
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
        [key, value],
    )?;
    Ok(())
}
