//! Core types used throughout the grepvec library.

use std::path::PathBuf;
use std::str::FromStr;

use serde::Serialize;

use crate::error::Error;

/// Blake3 hex digest of file content.
pub type DocHash = String;

/// A chunk of text extracted from a document.
#[derive(Debug)]
pub struct Chunk {
    pub doc_hash: DocHash,
    pub chunk_idx: u32,
    pub text: String,
    /// Byte offset in the original file where this chunk starts.
    pub byte_offset: u64,
}

/// A single search result returned to the caller.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub path: PathBuf,
    pub chunk_text: String,
    /// Vector similarity score.
    pub score: f32,
    /// Raw vector similarity score (same as `score` in vector-only mode).
    pub vec_score: f32,
    pub chunk_idx: u32,
    pub byte_offset: u64,
}

/// Options for a single search call.
#[derive(Debug)]
pub struct SearchOptions {
    pub query: String,
    pub top_k: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            query: String::new(),
            top_k: 10,
        }
    }
}

/// Which embedding model to use.
#[derive(Debug, Clone)]
pub enum ModelChoice {
    BgeLarge,
    MxbaiLarge,
    E5Large,
    GteLarge,
    ArcticLarge,
    ModernBertLarge,
    BgeM3,
    BgeSmall,
    External(String),
}

impl ModelChoice {
    /// Returns the embedding dimension for this model.
    pub fn dim(&self) -> usize {
        match self {
            Self::BgeSmall => 384,
            _ => 1024,
        }
    }
}

impl FromStr for ModelChoice {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "bge-large" => Ok(Self::BgeLarge),
            "mxbai-large" => Ok(Self::MxbaiLarge),
            "e5-large" => Ok(Self::E5Large),
            "gte-large" => Ok(Self::GteLarge),
            "arctic-large" => Ok(Self::ArcticLarge),
            "modernbert-large" => Ok(Self::ModernBertLarge),
            "bge-m3" => Ok(Self::BgeM3),
            "bge-small" => Ok(Self::BgeSmall),
            other => Err(Error::UnknownModel(other.to_string())),
        }
    }
}

/// Top-level configuration derived from CLI flags + defaults.
#[derive(Debug)]
pub struct Config {
    pub cache_dir: PathBuf,
    pub model: ModelChoice,
    /// TurboQuant quantization bit width (2 or 4).
    pub bit_width: u8,
    /// Target chunk size in bytes before splitting.
    pub chunk_size: usize,
    /// Number of results to return.
    pub top_k: usize,
    /// Skip the global cache entirely.
    pub no_cache: bool,
    /// Emit results as JSON.
    pub json_output: bool,
    /// Print file paths only (like grep -l).
    pub files_only: bool,
    /// Force re-embedding of all files.
    pub reindex: bool,
    /// Additional ignore pattern files (like .gitignore).
    pub ignore_files: Vec<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        let cache_dir = dirs_cache().join("grepvec");
        Self {
            cache_dir,
            model: ModelChoice::BgeLarge,
            bit_width: 4,
            chunk_size: 2048,
            top_k: 10,
            no_cache: false,
            json_output: false,
            files_only: false,
            reindex: false,
            ignore_files: Vec::new(),
        }
    }
}

fn dirs_cache() -> PathBuf {
    if let Ok(d) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(d)
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache")
    } else {
        PathBuf::from("/tmp")
    }
}
