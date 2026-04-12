<p align="center">
  <img src="logo.svg" alt="grepvec" width="480">
</p>

<p align="center">
  <strong>Grep-style vector similarity search.</strong><br>
  No index build step. No daemons. No config files.
</p>

---

Point it at a directory, give it a query, get results. Embeddings are computed on the fly, cached by content hash, and immediately queryable. Appending a file works on the next search with no rebuild.

Think of it as `ripgrep` for the case where you also want semantic similarity, not just substring matches.

## Install

**Prebuilt binaries** (macOS, Linux):

Download from [GitHub Releases](https://github.com/tylergibbs1/grepvec/releases/latest), extract, and add to your `PATH`:
```bash
# macOS (Apple Silicon)
curl -LO https://github.com/tylergibbs1/grepvec/releases/latest/download/grepvec-aarch64-apple-darwin.tar.gz
tar xzf grepvec-aarch64-apple-darwin.tar.gz
sudo mv grepvec /usr/local/bin/
```

**From source** (requires Rust 1.80+):
```bash
cargo install --git https://github.com/tylergibbs1/grepvec
```

On first run, the default embedding model (~400 MB) is downloaded and cached. Use `--model bge-small` (~30 MB) to start lighter.

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

# Pipe text from stdin:
$ cat incident.log | grepvec --stdin "auth failures around the Redis outage"
```

## Why not just ripgrep?

ripgrep finds exact text matches. grepvec finds **meaning**. When you don't know the exact words, ripgrep returns nothing:

```
Query: "how to handle too many requests"        rg: 0 results  grepvec: rate_limiting.md (0.625)
Query: "what to do when a service keeps failing" rg: 0 results  grepvec: circuit_breakers.md (0.626)
Query: "zero downtime releases"                  rg: 0 results  grepvec: deployment.md (0.698)
Query: "how to paginate API results efficiently"  rg: 0 results  grepvec: api_design.md (0.704)
```

Run `./benchmarks/vs_ripgrep.sh` to reproduce.

## How it works

grepvec embeds your query and documents with [fastembed](https://github.com/Anush008/fastembed-rs), quantizes with [TurboQuant](https://arxiv.org/abs/2504.19874) (via [turbovec](https://github.com/RyanCodrai/turbovec)), and does a SIMD brute-force scan to find the most similar chunks.

The key innovation from TurboQuant is that quantization codebooks are derived from math, not data. No training step means you can quantize one vector at a time with no setup -- which is what makes the grep model possible.

### Caching

Files are hashed with blake3. Embeddings are keyed by content hash, so:

- Renaming a file doesn't trigger re-embedding.
- Two projects sharing the same file share one embedding.
- Switching git branches only re-embeds changed files.
- mtime+size fast path skips rehashing when files haven't changed.
- Persistent `.tv` index is reloaded on repeat queries (no rebuild).
- Query embeddings are cached so repeated queries skip the model call.

Cache lives at `~/.cache/grepvec/` (override with `--cache-dir` or `GREPVEC_CACHE_DIR`).

## Usage

```
grepvec [OPTIONS] [QUERY] [PATHS]...
```

### Search flags

| Flag | Default | Description |
|------|---------|-------------|
| `-k, --top` | `10` | Number of results. |
| `-m, --model` | `bge-large` | Embedding model (see below). |
| `--bits` | `4` | TurboQuant bit width (2 or 4). |
| `-l, --files-only` | | Print paths only, like `grep -l`. |
| `--json` | | JSON output (auto-enabled when stdout is piped). |
| `--stdin` | | Read content from stdin instead of walking files. |
| `--ignore-file` | | Additional ignore pattern file (repeatable). |
| `--fields` | | Limit JSON output fields (e.g. `path,score`). |
| `--no-cache` | | Don't read or write the global cache. |
| `--reindex` | | Force re-embedding of all files. |
| `--chunk-size` | `2048` | Bytes per chunk before splitting. |
| `--dry-run` | | Validate without executing (for mutating commands). |

### Subcommands

```bash
# Explicit corpus management:
grepvec init PATH [--model NAME]    # Create a named corpus
grepvec add PATH FILE...            # Add files to a corpus
grepvec rm PATH DOC_ID...           # Remove documents
grepvec compact PATH                # Rewrite without tombstones
grepvec stats [PATH] [--json]       # Show corpus or cache stats

# Cache maintenance:
grepvec gc [--aggressive]            # Garbage-collect stale data
grepvec cache-info [--json]          # Show cache size and model

# Agent introspection:
grepvec schema COMMAND               # Dump command schema as JSON
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
  index.db              # sqlite: path -> content_hash, chunks, mtime
  vectors.tv            # persistent turbovec index (reloaded on repeat queries)
  embeddings/
    <blake3_hash>.emb   # raw f32 vectors, one file per document
  qcache/
    <query_hash>.qemb   # cached query embeddings
  models/
    bge-small-en-v1.5/  # fastembed model weights (auto-downloaded)
```

~2,200 lines of Rust. Dependencies: turbovec (vendored), fastembed, rusqlite, blake3, clap, ignore, indicatif.

## Limitations (v0.1)

- Single-node only. No distributed search.
- Brute-force scan, not HNSW. Designed for corpora up to ~1M documents.
- CPU embedding only (no GPU acceleration).
- English-optimized by default. Use `--model e5-large` or `--model bge-m3` for multilingual.
- External embedder (`--embed-cmd`) is not yet implemented.

## License

MIT OR Apache-2.0
