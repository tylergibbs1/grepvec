//! Embedding model abstraction over fastembed and external embedders.

use std::path::Path;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::error::{Error, Result};
use crate::types::ModelChoice;

fn to_fastembed(model: &ModelChoice) -> Option<EmbeddingModel> {
    match model {
        ModelChoice::BgeLarge => Some(EmbeddingModel::BGELargeENV15Q),
        ModelChoice::MxbaiLarge => Some(EmbeddingModel::MxbaiEmbedLargeV1),
        ModelChoice::E5Large => Some(EmbeddingModel::MultilingualE5Large),
        ModelChoice::GteLarge => Some(EmbeddingModel::GTELargeENV15),
        ModelChoice::ArcticLarge => Some(EmbeddingModel::SnowflakeArcticEmbedL),
        ModelChoice::ModernBertLarge => Some(EmbeddingModel::ModernBertEmbedLarge),
        ModelChoice::BgeM3 => Some(EmbeddingModel::BGEM3),
        ModelChoice::BgeSmall => Some(EmbeddingModel::BGESmallENV15Q),
        ModelChoice::External(_) => None,
    }
}

/// Wraps an embedding model for encoding text into vectors.
pub struct Embedder {
    inner: EmbedderInner,
    dim: usize,
}

enum EmbedderInner {
    Fastembed(Box<TextEmbedding>),
    #[expect(dead_code)]
    External(String),
}

impl Embedder {
    /// Create a new embedder for the given model. Downloads model weights on first use.
    pub fn new(model: &ModelChoice, cache_dir: &Path) -> Result<Self> {
        let dim = model.dim();
        let inner = match model {
            ModelChoice::External(cmd) => EmbedderInner::External(cmd.clone()),
            other => {
                let fe_model = to_fastembed(other)
                    .ok_or_else(|| Error::Embedding(format!("no fastembed model for {other:?}")))?;
                let opts = InitOptions::new(fe_model)
                    .with_cache_dir(cache_dir.join("models"))
                    .with_show_download_progress(true);
                let te = TextEmbedding::try_new(opts)
                    .map_err(|e| Error::Embedding(e.to_string()))?;
                EmbedderInner::Fastembed(Box::new(te))
            }
        };
        Ok(Self { inner, dim })
    }

    /// Embed a batch of texts into f32 vectors.
    pub fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match &mut self.inner {
            EmbedderInner::Fastembed(m) => m
                .embed(texts, None)
                .map_err(|e| Error::Embedding(e.to_string())),
            EmbedderInner::External(_) => Err(Error::ExternalEmbedderUnimplemented),
        }
    }

    /// Returns the embedding dimension for this model.
    pub fn dim(&self) -> usize {
        self.dim
    }
}
