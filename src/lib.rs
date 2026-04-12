//! `grepvec` -- vector similarity search library.
//!
//! Provides file walking, chunking, embedding, caching, and vector search
//! with no index build step and no background daemons.

pub mod chunk;
pub mod embed;
pub mod error;
pub mod search;
pub mod storage;
pub mod types;
pub mod walker;

use std::path::PathBuf;

use indicatif::{ProgressBar, ProgressStyle};

use crate::chunk::chunk_text;
use crate::embed::Embedder;
use crate::error::Result;
use crate::storage::cache::ImplicitCache;
use crate::storage::{StorageBackend, StoredChunk};
use crate::types::{Config, SearchResult};
use crate::walker::walk_paths;

/// Run an implicit-mode search: walk paths, index new/changed files, search.
pub fn implicit_search(
    query: &str,
    paths: &[PathBuf],
    config: &Config,
) -> Result<Vec<SearchResult>> {
    let files = walk_paths(paths, &config.ignore_files)?;
    if files.is_empty() {
        eprintln!("grepvec: no text files found in {paths:?}");
        return Ok(Vec::new());
    }

    let mut cache = ImplicitCache::open(&config.cache_dir)?;
    let mut embedder = Embedder::new(&config.model, &config.cache_dir)?;
    let dim = embedder.dim();

    let model_name = format!("{:?}", config.model);
    let cleared = cache.ensure_model(&model_name)?;
    if cleared {
        eprintln!("grepvec: model changed, cache cleared");
    }

    let new_count = index_files(&files, &mut cache, &mut embedder, config)?;
    if new_count > 0 {
        eprintln!(
            "[embedding {new_count} new file{}...]",
            if new_count == 1 { "" } else { "s" }
        );
    }

    let live_paths: Vec<PathBuf> = files.iter().map(|f| f.path.clone()).collect();
    let stale = cache.remove_stale(&live_paths)?;
    if stale > 0 {
        eprintln!(
            "[removed {stale} stale file{}]",
            if stale == 1 { "" } else { "s" }
        );
    }

    let all_chunks = cache.all_chunks_ordered()?;
    if all_chunks.is_empty() {
        return Ok(Vec::new());
    }

    let mut vec_index = build_vec_index(&cache, config, dim)?;
    let query_emb = load_or_embed_query(&mut embedder, &cache, query)?;

    Ok(vector_search(
        &mut vec_index,
        &query_emb,
        &all_chunks,
        config.top_k,
    ))
}

/// Search text piped via stdin. No caching, no persistence.
pub fn stdin_search(
    query: &str,
    stdin_text: &str,
    config: &Config,
) -> Result<Vec<SearchResult>> {
    let hash = blake3::hash(stdin_text.as_bytes()).to_hex().to_string();
    let chunks = chunk_text(stdin_text, &hash, config.chunk_size, config.chunk_size / 8);
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let mut embedder = Embedder::new(&config.model, &config.cache_dir)?;
    let dim = embedder.dim();

    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings = embedder.embed(&texts)?;

    let cache_dir = tempfile::tempdir().map_err(error::Error::Io)?;
    let mut cache = ImplicitCache::open(cache_dir.path())?;
    let model_name = format!("{:?}", config.model);
    cache.ensure_model(&model_name)?;

    let stdin_path = std::path::PathBuf::from("<stdin>");
    cache.upsert_document(&stdin_path, &hash, &chunks, &embeddings, 0, 0)?;

    let all_chunks = cache.all_chunks_ordered()?;
    if all_chunks.is_empty() {
        return Ok(Vec::new());
    }

    let vectors = load_embeddings_from_cache(&cache, dim)?;
    let mut vec_index =
        search::vector::VecIndex::build(&vectors, dim, config.bit_width as usize)?;
    let query_emb = embedder.embed(&[query.to_string()])?;

    Ok(vector_search(
        &mut vec_index,
        &query_emb[0],
        &all_chunks,
        config.top_k,
    ))
}

fn index_files(
    files: &[walker::WalkedFile],
    cache: &mut ImplicitCache,
    embedder: &mut Embedder,
    config: &Config,
) -> Result<usize> {
    let mut new_count = 0usize;
    let pb = ProgressBar::new(files.len() as u64);
    if let Ok(style) = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40} {pos}/{len} {msg}")
    {
        pb.set_style(style);
    }

    cache.begin_batch()?;

    for file in files {
        pb.set_message(
            file.path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
        );

        // mtime+size fast path: skip hashing if file hasn't changed.
        if !config.reindex
            && cache
                .check_freshness(&file.path, file.mtime_secs, file.file_size)
                .is_some()
        {
            pb.inc(1);
            continue;
        }

        let hash = blake3::hash(file.content.as_bytes()).to_hex().to_string();

        if !config.reindex && cache.has_hash(&hash)? {
            cache.upsert_document(
                &file.path, &hash, &[], &[], file.mtime_secs, file.file_size,
            )?;
            pb.inc(1);
            continue;
        }

        let chunks = chunk_text(&file.content, &hash, config.chunk_size, config.chunk_size / 8);
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = if texts.is_empty() {
            Vec::new()
        } else {
            embedder.embed(&texts)?
        };

        if cache.upsert_document(
            &file.path,
            &hash,
            &chunks,
            &embeddings,
            file.mtime_secs,
            file.file_size,
        )? {
            new_count += 1;
        }
        pb.inc(1);
    }

    cache.commit_batch()?;
    pb.finish_and_clear();
    Ok(new_count)
}

fn build_vec_index(
    cache: &ImplicitCache,
    config: &Config,
    dim: usize,
) -> Result<search::vector::VecIndex> {
    // Try loading persistent .tv index.
    if cache.is_tv_index_fresh()
        && let Ok(idx) = search::vector::VecIndex::load(&cache.tv_index_path())
    {
        return Ok(idx);
    }

    let vectors = load_embeddings_from_cache(cache, dim)?;
    let index = search::vector::VecIndex::build(&vectors, dim, config.bit_width as usize)?;

    // Persist for next query.
    if index.save(&cache.tv_index_path()).is_ok() {
        let _ = cache.mark_tv_index_built();
    }

    Ok(index)
}

fn load_embeddings_from_cache(cache: &ImplicitCache, dim: usize) -> Result<Vec<Vec<f32>>> {
    let conn = cache.connection();
    let mut stmt = conn.prepare(
        "SELECT c.id, d.hash, c.chunk_idx
         FROM chunks c JOIN documents d ON c.doc_id = d.id
         ORDER BY c.id ASC",
    )?;
    let rows: Vec<(i64, String, u32)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<std::result::Result<_, _>>()?;

    let mut vectors = Vec::with_capacity(rows.len());
    for (_, hash, chunk_idx) in &rows {
        let emb = cache.load_embedding(hash, *chunk_idx, dim)?;
        vectors.push(emb);
    }
    Ok(vectors)
}

/// Cache query embeddings by hash to avoid re-embedding repeated queries.
fn load_or_embed_query(
    emb: &mut Embedder,
    cache: &ImplicitCache,
    query: &str,
) -> Result<Vec<f32>> {
    let qcache_dir = cache.qcache_dir();
    let query_hash = blake3::hash(query.as_bytes()).to_hex().to_string();
    let qcache_path = qcache_dir.join(format!("{query_hash}.qemb"));

    if let Ok(bytes) = std::fs::read(&qcache_path)
        && bytes.len() == emb.dim() * 4
    {
        return Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect());
    }

    let embeddings = emb.embed(&[query.to_string()])?;
    let query_emb = embeddings.into_iter().next().unwrap_or_default();

    if std::fs::create_dir_all(&qcache_dir).is_ok() {
        let blob: Vec<u8> = query_emb.iter().flat_map(|f| f.to_le_bytes()).collect();
        let _ = std::fs::write(&qcache_path, &blob);
    }

    Ok(query_emb)
}

fn vector_search(
    index: &mut search::vector::VecIndex,
    query_emb: &[f32],
    all_chunks: &[StoredChunk],
    top_k: usize,
) -> Vec<SearchResult> {
    let k = top_k.min(index.len());
    let hits = index.search(query_emb, k);

    hits.into_iter()
        .map(|(pos, score)| {
            let chunk = &all_chunks[pos];
            SearchResult {
                path: chunk.doc_path.clone(),
                chunk_text: chunk.text.clone(),
                score,
                vec_score: score,
                chunk_idx: chunk.chunk_idx,
                byte_offset: chunk.byte_offset,
            }
        })
        .collect()
}
