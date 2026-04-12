//! TurboQuant vector index wrapper.

use std::path::Path;

use crate::error::{Error, Result};
use turbovec::TurboQuantIndex;

/// Wrapper around `TurboQuantIndex` for building and searching.
pub struct VecIndex {
    inner: TurboQuantIndex,
}

impl VecIndex {
    /// Build an index from a set of f32 vectors.
    pub fn build(vectors: &[Vec<f32>], dim: usize, bit_width: usize) -> Result<Self> {
        if vectors.is_empty() {
            return Err(Error::VecIndex("cannot build from 0 vectors".into()));
        }
        let mut index = TurboQuantIndex::new(dim, bit_width);
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        index.add(&flat);
        Ok(Self { inner: index })
    }

    /// Search for the top-k nearest neighbors of a query vector.
    /// Returns `(index_in_vectors, score)` pairs.
    pub fn search(&mut self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let results = self.inner.search(query, k);
        let scores = results.scores_for_query(0);
        let indices = results.indices_for_query(0);

        indices
            .iter()
            .zip(scores.iter())
            .map(|(&idx, &score)| (idx as usize, score))
            .collect()
    }

    /// Save the index to disk.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.inner
            .write(path)
            .map_err(|e| Error::VecIndex(format!("write to {}: {e}", path.display())))?;
        Ok(())
    }

    /// Load an index from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let inner = TurboQuantIndex::load(path)
            .map_err(|e| Error::VecIndex(format!("load from {}: {e}", path.display())))?;
        Ok(Self { inner })
    }

    /// Returns the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
