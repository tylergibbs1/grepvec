#![allow(warnings)]
//! TurboQuant implementation for vector search.
//!
//! Compresses high-dimensional vectors to 2-4 bits per coordinate with
//! near-optimal distortion. Data-oblivious — no training required.
//!
//! ```rust
//! use turbovec::TurboQuantIndex;
//!
//! let mut index = TurboQuantIndex::new(1536, 4);
//! index.add(&vectors);
//! let results = index.search(&queries, 10);
//! index.write("index.tv").unwrap();
//! let loaded = TurboQuantIndex::load("index.tv").unwrap();
//! ```

pub mod codebook;
pub mod encode;
pub mod io;
pub mod pack;
pub mod rotation;
pub mod search;

use std::path::Path;

const ROTATION_SEED: u64 = 42;
const BLOCK: usize = 32;
const FLUSH_EVERY: usize = 256;

pub struct TurboQuantIndex {
    dim: usize,
    bit_width: usize,
    n_vectors: usize,
    packed_codes: Vec<u8>,
    norms: Vec<f32>,
    /// Rotated unit vectors stored for reranking (n_vectors * dim, f32).
    /// Enables exact dot-product reranking of top candidates.
    rotated_vectors: Vec<f32>,
    // Cached for search (lazily computed)
    blocked: Option<Vec<u8>>,
    n_blocks: usize,
    centroids: Option<Vec<f32>>,
    rotation: Option<Vec<f32>>,
}

pub struct SearchResults {
    pub scores: Vec<f32>,
    pub indices: Vec<i64>,
    pub nq: usize,
    pub k: usize,
}

impl SearchResults {
    pub fn scores_for_query(&self, qi: usize) -> &[f32] {
        &self.scores[qi * self.k..(qi + 1) * self.k]
    }

    pub fn indices_for_query(&self, qi: usize) -> &[i64] {
        &self.indices[qi * self.k..(qi + 1) * self.k]
    }
}

impl TurboQuantIndex {
    pub fn new(dim: usize, bit_width: usize) -> Self {
        assert!(
            bit_width >= 2 && bit_width <= 4,
            "bit_width must be 2, 3, or 4"
        );
        assert!(dim % 8 == 0, "dim must be a multiple of 8");

        Self {
            dim,
            bit_width,
            n_vectors: 0,
            packed_codes: Vec::new(),
            norms: Vec::new(),
            rotated_vectors: Vec::new(),
            blocked: None,
            n_blocks: 0,
            centroids: None,
            rotation: None,
        }
    }

    pub fn add(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        assert_eq!(
            vectors.len(),
            n * self.dim,
            "vectors length must be a multiple of dim"
        );

        let rotation = self.ensure_rotation();
        let (boundaries, _) = codebook::codebook(self.bit_width, self.dim);
        let (packed, norms, rotated) = encode::encode(vectors, n, self.dim, &rotation, &boundaries, self.bit_width);

        if self.n_vectors == 0 {
            self.packed_codes = packed;
            self.norms = norms;
            self.rotated_vectors = rotated;
        } else {
            self.packed_codes.extend_from_slice(&packed);
            self.norms.extend_from_slice(&norms);
            self.rotated_vectors.extend_from_slice(&rotated);
        }
        self.n_vectors += n;
        self.blocked = None; // invalidate cache
    }

    pub fn search(&mut self, queries: &[f32], k: usize) -> SearchResults {
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);

        self.ensure_cached();

        let rotation = self.rotation.as_ref().unwrap();
        let blocked = self.blocked.as_ref().unwrap();
        let centroids = self.centroids.as_ref().unwrap();
        let k = k.min(self.n_vectors);

        // Oversample by 128x, then rerank with exact dot product scores.
        let oversample = (128 * k).min(self.n_vectors);

        let (raw_scores, raw_indices) = search::search(
            queries,
            nq,
            rotation,
            blocked,
            centroids,
            &self.norms,
            self.bit_width,
            self.dim,
            self.n_vectors,
            self.n_blocks,
            oversample,
        );

        // Rotate queries for reranking (exact dot product with stored rotated vectors)
        let q_rot: Vec<Vec<f32>> = (0..nq)
            .map(|qi| {
                let q_row = &queries[qi * self.dim..(qi + 1) * self.dim];
                let mut rotated = vec![0.0f32; self.dim];
                for j in 0..self.dim {
                    let r_row = &rotation[j * self.dim..(j + 1) * self.dim];
                    let mut dot = 0.0f32;
                    for d in 0..self.dim {
                        dot += q_row[d] * r_row[d];
                    }
                    rotated[j] = dot;
                }
                rotated
            })
            .collect();

        // Rerank using exact dot product with stored rotated unit vectors
        let mut all_scores = Vec::with_capacity(nq * k);
        let mut all_indices = Vec::with_capacity(nq * k);

        for qi in 0..nq {
            let cand_indices = &raw_indices[qi * oversample..(qi + 1) * oversample];

            let mut reranked: Vec<(f32, i64)> = cand_indices
                .iter()
                .filter(|&&idx| idx >= 0)
                .map(|&idx| {
                    let vi = idx as usize;
                    let v_rot = &self.rotated_vectors[vi * self.dim..(vi + 1) * self.dim];
                    let dot: f32 = q_rot[qi].iter().zip(v_rot.iter()).map(|(a, b)| a * b).sum();
                    (dot * self.norms[vi], idx)
                })
                .collect();

            reranked.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            reranked.truncate(k);

            let pad = k.saturating_sub(reranked.len());
            all_scores.extend(reranked.iter().map(|(s, _)| *s));
            all_scores.extend(std::iter::repeat(f32::NEG_INFINITY).take(pad));
            all_indices.extend(reranked.iter().map(|(_, i)| *i));
            all_indices.extend(std::iter::repeat(0i64).take(pad));
        }

        SearchResults {
            scores: all_scores,
            indices: all_indices,
            nq,
            k,
        }
    }

    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        io::write(path, self.bit_width, self.dim, self.n_vectors, &self.packed_codes, &self.norms)
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let (bit_width, dim, n_vectors, packed_codes, norms) = io::load(path)?;
        Ok(Self {
            dim,
            bit_width,
            n_vectors,
            packed_codes,
            norms,
            rotated_vectors: Vec::new(), // Not persisted; reranking falls back to centroid-based
            blocked: None,
            n_blocks: 0,
            centroids: None,
            rotation: None,
        })
    }

    pub fn len(&self) -> usize {
        self.n_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    fn ensure_rotation(&mut self) -> Vec<f32> {
        if self.rotation.is_none() {
            self.rotation = Some(rotation::make_rotation_matrix(self.dim));
        }
        self.rotation.clone().unwrap()
    }

    fn ensure_cached(&mut self) {
        if self.rotation.is_none() {
            self.rotation = Some(rotation::make_rotation_matrix(self.dim));
        }
        if self.centroids.is_none() {
            let (_, c) = codebook::codebook(self.bit_width, self.dim);
            self.centroids = Some(c);
        }
        if self.blocked.is_none() {
            let (blocked, n_blocks) =
                pack::repack(&self.packed_codes, self.n_vectors, self.bit_width, self.dim);
            self.blocked = Some(blocked);
            self.n_blocks = n_blocks;
        }
    }
}
