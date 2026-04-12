//! File discovery with `.gitignore` support via the `ignore` crate.

use std::path::{Path, PathBuf};

use crate::error::Result;
use ignore::WalkBuilder;

/// A discovered file: its path, UTF-8 text content, and metadata for freshness checks.
#[derive(Debug)]
pub struct WalkedFile {
    pub path: PathBuf,
    pub content: String,
    /// File modification time as seconds since Unix epoch.
    pub mtime_secs: i64,
    /// File size in bytes.
    pub file_size: i64,
}

/// Walk `paths`, respecting `.gitignore`, yielding text files.
/// `ignore_files` is an optional list of additional ignore pattern files.
pub fn walk_paths(paths: &[PathBuf], ignore_files: &[PathBuf]) -> Result<Vec<WalkedFile>> {
    let mut files = Vec::new();

    for path in paths {
        if path.is_file() {
            if let Some(f) = try_read_text_file(path) {
                files.push(f);
            }
        } else if path.is_dir() {
            let mut builder = WalkBuilder::new(path);
            builder
                .hidden(true)
                .git_ignore(true)
                .git_global(true)
                .git_exclude(true);

            for ignore_file in ignore_files {
                builder.add_ignore(ignore_file);
            }

            let walker = builder.build();

            for entry in walker {
                let entry = entry?;
                if entry.file_type().is_some_and(|ft| ft.is_file())
                    && let Some(f) = try_read_text_file(entry.path())
                {
                    files.push(f);
                }
            }
        }
    }

    Ok(files)
}

/// Try to read a file as UTF-8 text. Returns `None` for binary or empty files.
fn try_read_text_file(path: &Path) -> Option<WalkedFile> {
    let metadata = std::fs::metadata(path).ok()?;
    let file_size = metadata.len() as i64;
    let mtime_secs = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs() as i64);

    let bytes = std::fs::read(path).ok()?;

    // Quick binary check: null byte in the first 8KB means binary.
    let check_len = bytes.len().min(8192);
    if bytes[..check_len].contains(&0) {
        return None;
    }

    let content = String::from_utf8(bytes).ok()?;
    if content.trim().is_empty() {
        return None;
    }

    Some(WalkedFile {
        path: path.to_path_buf(),
        content,
        mtime_secs,
        file_size,
    })
}
