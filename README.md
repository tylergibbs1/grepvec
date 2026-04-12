<p align="center">
  <img src="logo.svg" alt="grepvec" width="480">
</p>

<p align="center">
  <strong>Grep-style hybrid vector + fulltext search.</strong><br>
  No index build step. No daemons. No config files.
</p>

---

Point it at a directory, give it a query, get results. Embeddings are computed on the fly, cached by content hash, and immediately queryable. Appending a file works on the next search with no rebuild.

Think of it as `ripgrep` for the case where you also want semantic similarity, not just substring matches.

## Install

```
cargo install --git https://github.com/YOUR_USER/grepvec
```

Requires Rust 1.80+. On first run, the default embedding model (~400 MB) is downloaded and cached.

## Quick start

```bash
# Search your notes (embeds on first run, cached after):
$ grepvec "things I said about rate limiting" ~/notes
[embedding 342 new files...]
notes/2024-03-infra.md:12  0.891  I keep thinking about rate limiting at the gateway...
notes/backend-ideas.md:44  0.812  rate limiting is really a backpressure problem

# Second run is instant (nothing changed):
$ grepvec "circuit breakers" ~/notes
notes/2024-03-infra.md:89  0.903  circuit breakers vs retries: the tradeoff is...

# Add a new file, only that one gets embedded:
$ echo "new thought about queues" > ~/notes/queues.md
$ grepvec "queues" ~/notes
[embedding 1 new file...]
notes/queues.md:1  0.987  new thought about queues
```

## How it works

grepvec combines two search methods in one pass:

1. **Vector search** -- embeds your query and documents with [fastembed](https://github.com/Anush008/fastembed-rs), quantizes with [TurboQuant](https://arxiv.org/abs/2504.19874) (via [turbovec](https://github.com/RyanCodrai/turbovec)), and does a SIMD brute-force scan.
2. **BM25 fulltext** -- classic keyword scoring over the same corpus.

Scores are normalized and merged: `final = alpha * vector + (1 - alpha) * bm25`. Default alpha is 0.5. Tune with `--alpha`.

The key innovation from TurboQuant is that quantization codebooks are derived from math, not data. No training step means you can quantize one vector at a time with no setup -- which is what makes the grep model possible.

### Caching

Files are hashed with blake3. Embeddings are keyed by content hash, so:

- Renaming a file doesn't trigger re-embedding.
- Two projects sharing the same file share one embedding.
- Switching git branches only re-embeds changed files.

Cache lives at `~/.cache/grepvec/` (override with `--cache-dir` or `GREPVEC_CACHE_DIR`).

## Usage

```
grepvec [OPTIONS] [QUERY] [PATHS]...
```

### Search flags

| Flag | Default | Description |
|------|---------|-------------|
| `-a, --alpha` | `0.5` | Hybrid weight. `0` = pure BM25, `1` = pure vector. |
| `-k, --top` | `10` | Number of results. |
| `-m, --model` | `bge-large` | Embedding model (see below). |
| `--bits` | `4` | TurboQuant bit width (2 or 4). |
| `-l, --files-only` | | Print paths only, like `grep -l`. |
| `--json` | | JSON output. |
| `--no-embed` | | Text-only mode (BM25 only, skip embedding). |
| `--no-cache` | | Don't read or write the global cache. |
| `--reindex` | | Force re-embedding of all files. |
| `--chunk-size` | `2048` | Bytes per chunk before splitting. |

### Subcommands

```bash
# Explicit corpus management:
grepvec init PATH [--model NAME]    # Create a named corpus
grepvec add PATH FILE...            # Add files to a corpus
grepvec rm PATH DOC_ID...           # Remove documents
grepvec compact PATH                # Rewrite without tombstones
grepvec stats [PATH]                # Show corpus or cache stats

# Cache maintenance:
grepvec gc [--aggressive]            # Garbage-collect stale data
grepvec cache-info                   # Show cache size and model
```

### Exit codes

Matches grep: `0` = matches found, `1` = no matches, `2` = error.

## Models

All models are downloaded automatically by fastembed on first use.

| `--model` | Dimensions | Size | Notes |
|-----------|-----------|------|-------|
| `bge-large` | 1024 | ~400 MB | Default. English. Strong retrieval. |
| `mxbai-large` | 1024 | ~400 MB | English. Slightly stronger on MTEB. |
| `e5-large` | 1024 | ~400 MB | Multilingual (~100 languages). |
| `gte-large` | 1024 | ~400 MB | 8k context. Good for long documents. |
| `arctic-large` | 1024 | ~400 MB | Retrieval-tuned. |
| `modernbert-large` | 1024 | ~400 MB | Newer architecture, faster, 8k context. |
| `bge-m3` | 1024 | ~400 MB | Multilingual, 8k context. |
| `bge-small` | 384 | ~30 MB | Tiny. For trying out or constrained environments. |

## Library usage

grepvec is also a Rust crate:

```rust
use grepvec::{implicit_search, types::{Config, ModelChoice}};

let config = Config {
    model: ModelChoice::BgeSmall,
    alpha: 0.7,
    top_k: 5,
    ..Config::default()
};

let results = implicit_search("retry logic", &["./src".into()], &config)?;
for r in &results {
    println!("{} {:.3} {}", r.path.display(), r.score, r.chunk_text.lines().next().unwrap_or(""));
}
```

## Architecture

```
~/.cache/grepvec/
  index.db              # sqlite: path -> content_hash, chunks, BM25 terms
  embeddings/
    <blake3_hash>.emb   # raw f32 vectors, one file per document
  models/
    bge-small-en-v1.5/  # fastembed model weights (auto-downloaded)
```

~1,850 lines of Rust. Dependencies: turbovec, fastembed, rusqlite, blake3, clap, ignore, indicatif.

## Limitations (v0.1)

- Single-node only. No distributed search.
- Brute-force scan, not HNSW. Designed for corpora up to ~1M documents.
- CPU embedding only (no GPU acceleration).
- English-optimized by default. Use `--model e5-large` or `--model bge-m3` for multilingual.
- No incremental vector index updates -- the turbovec index is rebuilt when documents change.
- External embedder (`--embed-cmd`) is not yet implemented.

## License

MIT OR Apache-2.0
