//! Chunking logic for splitting documents into overlapping text segments.

use crate::types::{Chunk, DocHash};

/// Split text into overlapping chunks at paragraph boundaries.
///
/// `max_bytes` is the target size per chunk in bytes. `overlap_bytes` is how
/// many bytes of overlap between consecutive chunks.
pub fn chunk_text(
    text: &str,
    doc_hash: &str,
    max_bytes: usize,
    overlap_bytes: usize,
) -> Vec<Chunk> {
    if text.is_empty() {
        return Vec::new();
    }

    let hash: DocHash = doc_hash.to_owned();

    if text.len() <= max_bytes {
        return vec![Chunk {
            doc_hash: hash,
            chunk_idx: 0,
            text: text.to_string(),
            byte_offset: 0,
        }];
    }

    let mut chunks = Vec::new();
    let bytes = text.as_bytes();
    let mut start: usize = 0;

    while start < bytes.len() {
        let raw_end = (start + max_bytes).min(bytes.len());
        // Snap to a char boundary (walk backwards if we land inside a multi-byte char).
        let end = snap_to_char_boundary(text, raw_end);
        if end <= start {
            break;
        }
        let break_at = find_break_point(text, start, end);

        let slice = &text[start..break_at];
        if !slice.trim().is_empty() {
            chunks.push(Chunk {
                doc_hash: hash.clone(),
                chunk_idx: chunks.len() as u32,
                text: slice.to_string(),
                byte_offset: start as u64,
            });
        }

        // Advance. Apply overlap, but never go backwards.
        let next = break_at.saturating_sub(overlap_bytes);
        let next = next.max(start + 1);
        start = snap_to_char_boundary(text, next);
    }

    chunks
}

/// Snap a byte offset to the nearest char boundary at or before `pos`.
fn snap_to_char_boundary(text: &str, pos: usize) -> usize {
    let mut p = pos.min(text.len());
    while p > 0 && !text.is_char_boundary(p) {
        p -= 1;
    }
    p
}

/// Find the best break point in `text[start..end]`.
/// Prefers double-newline, then single newline, then `end`.
fn find_break_point(text: &str, start: usize, end: usize) -> usize {
    let segment = &text[start..end];

    if let Some(pos) = segment.rfind("\n\n") {
        let break_at = start + pos + 2;
        if break_at > start {
            return break_at;
        }
    }

    if let Some(pos) = segment.rfind('\n') {
        let break_at = start + pos + 1;
        if break_at > start {
            return break_at;
        }
    }

    end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_returns_single_chunk_when_text_fits() {
        let chunks = chunk_text("hello world", "abc", 2048, 128);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
        assert_eq!(chunks[0].byte_offset, 0);
    }

    #[test]
    fn chunk_text_returns_empty_vec_for_empty_input() {
        let chunks = chunk_text("", "abc", 2048, 128);
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_text_splits_on_paragraph_boundary() {
        let text = "paragraph one\n\nparagraph two\n\nparagraph three";
        let chunks = chunk_text(text, "abc", 20, 5);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn chunk_text_handles_multibyte_utf8() {
        // The en-dash '–' is 3 bytes (U+2013). Chunk boundaries must not split it.
        let text = "HTRF – A Technology Review\n\nSecond paragraph here.";
        let chunks = chunk_text(text, "abc", 20, 5);
        assert!(!chunks.is_empty());
        for c in &chunks {
            // Every chunk text must be valid UTF-8 (would panic otherwise).
            assert!(!c.text.trim().is_empty());
        }
    }

    #[test]
    fn chunk_text_assigns_sequential_indices() {
        let text = "a\n\nb\n\nc\n\nd\n\ne";
        let chunks = chunk_text(text, "abc", 4, 1);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.chunk_idx, i as u32);
        }
    }
}
