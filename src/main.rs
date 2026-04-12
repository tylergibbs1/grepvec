// Force-link BLAS (needed for turbovec's ndarray).
#[cfg(target_os = "macos")]
extern crate accelerate_src;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use grepvec::storage::StorageBackend;

/// Grep-style vector similarity search.
#[derive(Parser)]
#[command(name = "grepvec", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Search query.
    query: Option<String>,

    /// Paths to search (default: current directory).
    #[arg(trailing_var_arg = true)]
    paths: Vec<PathBuf>,

    /// Number of results to return.
    #[arg(short = 'k', long = "top", default_value = "10")]
    top_k: usize,

    /// Embedding model.
    #[arg(short, long, default_value = "bge-large")]
    model: String,

    /// Quantization bit width (2 or 4).
    #[arg(long, default_value = "4")]
    bits: u8,

    /// Output results as JSON (auto-enabled when stdout is not a TTY).
    #[arg(long)]
    json: bool,

    /// Print file paths only (like grep -l).
    #[arg(short = 'l', long = "files-only")]
    files_only: bool,

    /// Don't use the global cache.
    #[arg(long)]
    no_cache: bool,

    /// Force re-embedding of all files.
    #[arg(long)]
    reindex: bool,

    /// Cache directory.
    #[arg(long, env = "GREPVEC_CACHE_DIR")]
    cache_dir: Option<PathBuf>,

    /// Chunk size in bytes.
    #[arg(long, default_value = "2048")]
    chunk_size: usize,

    /// Read content from stdin instead of walking files.
    #[arg(long)]
    stdin: bool,

    /// Additional ignore pattern file (like .gitignore).
    #[arg(long = "ignore-file")]
    ignore_file: Vec<PathBuf>,

    /// Limit output fields (comma-separated: path,score,chunk_text,vec_score,chunk_idx,byte_offset).
    #[arg(long, global = true)]
    fields: Option<String>,

    /// Validate the command without executing (for mutating operations).
    #[arg(long, global = true)]
    dry_run: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Create an explicit corpus.
    Init {
        path: PathBuf,
        #[arg(short, long, default_value = "bge-large")]
        model: String,
    },
    /// Add files to an explicit corpus.
    Add {
        path: PathBuf,
        files: Vec<PathBuf>,
    },
    /// Tombstone documents in an explicit corpus.
    Rm {
        path: PathBuf,
        doc_ids: Vec<String>,
    },
    /// Rewrite a corpus without tombstones.
    Compact { path: PathBuf },
    /// Show corpus or cache statistics.
    Stats {
        path: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Garbage-collect stale cache blobs.
    Gc {
        #[arg(long)]
        aggressive: bool,
    },
    /// Show cache size and info.
    CacheInfo {
        #[arg(long)]
        json: bool,
    },
    /// Dump machine-readable schema for a command.
    Schema { command: String },
}

fn main() {
    if let Err(e) = run() {
        emit_error(&e);
        process::exit(2);
    }
}

fn emit_error(err: &anyhow::Error) {
    if std::io::stderr().is_terminal() {
        eprintln!("grepvec: {err:#}");
    } else {
        let json = serde_json::json!({
            "error": true,
            "code": error_code(err),
            "message": format!("{err:#}"),
        });
        eprintln!("{json}");
    }
}

fn error_code(err: &anyhow::Error) -> &'static str {
    if let Some(e) = err.downcast_ref::<grepvec::error::Error>() {
        match e {
            grepvec::error::Error::UnknownModel(_) => "UNKNOWN_MODEL",
            grepvec::error::Error::Embedding(_) => "EMBEDDING_FAILED",
            grepvec::error::Error::ExternalEmbedderUnimplemented => "NOT_IMPLEMENTED",
            grepvec::error::Error::Storage(_) => "STORAGE_ERROR",
            grepvec::error::Error::Sqlite(_) => "DATABASE_ERROR",
            grepvec::error::Error::Io(_) => "IO_ERROR",
            grepvec::error::Error::Walk(_) => "WALK_ERROR",
            grepvec::error::Error::VecIndex(_) => "INDEX_ERROR",
            grepvec::error::Error::Other(_) => "UNKNOWN",
        }
    } else {
        "UNKNOWN"
    }
}

fn resolve_cache_dir(cli_cache_dir: Option<&PathBuf>) -> PathBuf {
    cli_cache_dir
        .cloned()
        .unwrap_or_else(|| grepvec::types::Config::default().cache_dir)
}

fn should_json(explicit_flag: bool) -> bool {
    explicit_flag || !std::io::stdout().is_terminal()
}

fn validate_path(path: &std::path::Path) -> Result<PathBuf> {
    let canonical = std::fs::canonicalize(path)
        .or_else(|_| Ok::<PathBuf, std::io::Error>(path.to_path_buf()))?;
    let path_str = canonical.to_string_lossy();
    if path_str.contains("..") {
        anyhow::bail!("path traversal rejected: {path_str}");
    }
    Ok(canonical)
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let cache_dir = resolve_cache_dir(cli.cache_dir.as_ref());

    match &cli.command {
        Some(Commands::Init { path, model }) => {
            let path = validate_path(path)?;
            cmd_init(&path, model, cli.dry_run)
        }
        Some(Commands::Add { path, files }) => {
            let path = validate_path(path)?;
            cmd_add(&path, files, &cache_dir, cli.chunk_size, cli.dry_run)
        }
        Some(Commands::Rm { path, doc_ids }) => {
            let path = validate_path(path)?;
            cmd_rm(&path, doc_ids, cli.dry_run)
        }
        Some(Commands::Compact { path }) => {
            let path = validate_path(path)?;
            cmd_compact(&path, cli.dry_run)
        }
        Some(Commands::Stats { path, json }) => {
            cmd_stats(path.as_deref(), &cache_dir, should_json(*json))
        }
        Some(Commands::Gc { aggressive }) => cmd_gc(&cache_dir, *aggressive, cli.dry_run),
        Some(Commands::CacheInfo { json }) => cmd_cache_info(&cache_dir, should_json(*json)),
        Some(Commands::Schema { command }) => cmd_schema(command),
        None => cmd_search(&cli, &cache_dir),
    }
}

fn cmd_init(path: &std::path::Path, model: &str, dry_run: bool) -> Result<()> {
    let model_choice = grepvec::types::ModelChoice::from_str(model)?;
    let model_name = format!("{model_choice:?}");
    if dry_run {
        eprintln!("dry-run: would create corpus at {} (model: {model})", path.display());
        return Ok(());
    }
    grepvec::storage::corpus::NamedCorpus::create(path, &model_name)?;
    eprintln!("created corpus at {} (model: {model})", path.display());
    Ok(())
}

fn cmd_add(
    path: &std::path::Path,
    files: &[PathBuf],
    cache_dir: &std::path::Path,
    chunk_size: usize,
    dry_run: bool,
) -> Result<()> {
    let mut corpus = grepvec::storage::corpus::NamedCorpus::open(path)?;
    let model_name = corpus.model_name()?;
    let model_choice = grepvec::types::ModelChoice::from_str(&model_name)
        .context(format!("unknown model '{model_name}' in corpus"))?;
    let walked = grepvec::walker::walk_paths(files, &[])?;
    if dry_run {
        eprintln!(
            "dry-run: would add {} text file(s) to corpus at {}",
            walked.len(),
            path.display()
        );
        return Ok(());
    }
    let mut embedder = grepvec::embed::Embedder::new(&model_choice, cache_dir)?;
    let mut added = 0;
    for file in &walked {
        let hash = blake3::hash(file.content.as_bytes()).to_hex().to_string();
        let chunks = grepvec::chunk::chunk_text(&file.content, &hash, chunk_size, chunk_size / 8);
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = if texts.is_empty() {
            Vec::new()
        } else {
            embedder.embed(&texts)?
        };
        if corpus.upsert_document(&file.path, &hash, &chunks, &embeddings, file.mtime_secs, file.file_size)? {
            added += 1;
        }
    }
    eprintln!("added {added} file(s) to corpus at {}", path.display());
    Ok(())
}

fn cmd_rm(path: &std::path::Path, doc_ids: &[String], dry_run: bool) -> Result<()> {
    if dry_run {
        eprintln!("dry-run: would remove {} doc(s) from corpus at {}", doc_ids.len(), path.display());
        return Ok(());
    }
    let corpus = grepvec::storage::corpus::NamedCorpus::open(path)?;
    let conn = corpus.connection();
    for id in doc_ids {
        conn.execute("DELETE FROM documents WHERE path = ?1", [id])?;
    }
    eprintln!("removed {} doc(s) from corpus at {}", doc_ids.len(), path.display());
    Ok(())
}

fn cmd_compact(path: &std::path::Path, dry_run: bool) -> Result<()> {
    if dry_run {
        eprintln!("dry-run: would compact corpus at {}", path.display());
        return Ok(());
    }
    let corpus = grepvec::storage::corpus::NamedCorpus::open(path)?;
    corpus.compact()?;
    eprintln!("compacted corpus at {}", path.display());
    Ok(())
}

fn cmd_stats(corpus_path: Option<&std::path::Path>, cache_dir: &std::path::Path, json: bool) -> Result<()> {
    if let Some(p) = corpus_path {
        let corpus = grepvec::storage::corpus::NamedCorpus::open(p)?;
        let stats = corpus.stats()?;
        if json {
            println!("{}", serde_json::json!({"type":"corpus","path":p.display().to_string(),"documents":stats.doc_count,"chunks":stats.chunk_count,"disk_bytes":stats.disk_bytes}));
        } else {
            println!("corpus: {}\n  documents: {}\n  chunks:    {}\n  disk:      {:.1} MB", p.display(), stats.doc_count, stats.chunk_count, stats.disk_bytes as f64 / 1_048_576.0);
        }
    } else if cache_dir.exists() {
        let mut total = 0u64;
        for entry in std::fs::read_dir(cache_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() { total += entry.metadata()?.len(); }
        }
        let emb_dir = cache_dir.join("embeddings");
        if emb_dir.exists() {
            for entry in std::fs::read_dir(&emb_dir)? {
                total += entry?.metadata()?.len();
            }
        }
        if json {
            println!("{}", serde_json::json!({"type":"cache","path":cache_dir.display().to_string(),"disk_bytes":total}));
        } else {
            println!("cache: {}\n  disk: {:.1} MB", cache_dir.display(), total as f64 / 1_048_576.0);
        }
    } else if json {
        println!("{}", serde_json::json!({"type":"cache","path":cache_dir.display().to_string(),"exists":false}));
    } else {
        println!("cache: {} (not found)", cache_dir.display());
    }
    Ok(())
}

fn cmd_gc(cache_dir: &std::path::Path, aggressive: bool, dry_run: bool) -> Result<()> {
    let emb_dir = cache_dir.join("embeddings");
    if !emb_dir.exists() { eprintln!("no cache to gc"); return Ok(()); }
    if dry_run {
        eprintln!("dry-run: would {} cached data at {}", if aggressive { "clear all" } else { "gc" }, cache_dir.display());
        return Ok(());
    }
    if aggressive {
        std::fs::remove_dir_all(&emb_dir)?;
        std::fs::create_dir_all(&emb_dir)?;
        let db = cache_dir.join("index.db");
        if db.exists() { std::fs::remove_file(&db)?; }
        let tv = cache_dir.join("vectors.tv");
        if tv.exists() { std::fs::remove_file(&tv)?; }
        eprintln!("cleared all cached data");
    } else {
        eprintln!("gc: run with --aggressive to clear all. Incremental gc not yet implemented.");
    }
    Ok(())
}

fn cmd_cache_info(cache_dir: &std::path::Path, json: bool) -> Result<()> {
    let db_path = cache_dir.join("index.db");
    if cache_dir.exists() && db_path.exists() {
        let conn = rusqlite::Connection::open(&db_path)?;
        let doc_count: i64 = conn.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;
        let chunk_count: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
        let model: String = conn.query_row("SELECT value FROM meta WHERE key = 'model'", [], |row| row.get(0)).unwrap_or_else(|_| "(unknown)".to_string());
        if json {
            println!("{}", serde_json::json!({"cache_dir":cache_dir.display().to_string(),"model":model,"documents":doc_count,"chunks":chunk_count}));
        } else {
            println!("cache dir: {}\n  model:     {model}\n  documents: {doc_count}\n  chunks:    {chunk_count}", cache_dir.display());
        }
    } else if json {
        println!("{}", serde_json::json!({"cache_dir":cache_dir.display().to_string(),"initialized":false}));
    } else {
        println!("cache dir: {}\n  (not initialized)", cache_dir.display());
    }
    Ok(())
}

fn cmd_schema(command: &str) -> Result<()> {
    let schema = match command {
        "search" => serde_json::json!({
            "command": "grepvec QUERY [PATH...]",
            "description": "Vector similarity search over files",
            "args": {
                "query": {"type": "string", "required": true},
                "paths": {"type": "array", "items": "path", "default": ["."]}
            },
            "flags": {
                "--top": {"type": "integer", "default": 10},
                "--model": {"type": "string", "default": "bge-large", "enum": ["bge-large","mxbai-large","e5-large","gte-large","arctic-large","modernbert-large","bge-m3","bge-small"]},
                "--bits": {"type": "integer", "default": 4, "enum": [2, 4]},
                "--json": {"type": "boolean", "default": false},
                "--fields": {"type": "string"},
                "--files-only": {"type": "boolean", "default": false},
                "--stdin": {"type": "boolean", "default": false},
                "--ignore-file": {"type": "path", "repeatable": true},
                "--no-cache": {"type": "boolean", "default": false},
                "--reindex": {"type": "boolean", "default": false},
                "--chunk-size": {"type": "integer", "default": 2048},
            },
            "exit_codes": {"0": "matches found", "1": "no matches", "2": "error"},
        }),
        "init" | "add" | "rm" | "gc" => serde_json::json!({"command": command, "mutating": true, "flags": {"--dry-run": {"type": "boolean"}}}),
        "stats" | "cache-info" => serde_json::json!({"command": command, "flags": {"--json": {"type": "boolean"}}}),
        other => anyhow::bail!("unknown command: {other}"),
    };
    println!("{}", serde_json::to_string_pretty(&schema)?);
    Ok(())
}

fn cmd_search(cli: &Cli, cache_dir: &std::path::Path) -> Result<()> {
    let query = cli.query.as_deref()
        .ok_or_else(|| anyhow::anyhow!("missing query argument\nusage: grepvec QUERY [PATH...]"))?;
    let use_json = should_json(cli.json);
    let config = grepvec::types::Config {
        cache_dir: cache_dir.to_path_buf(),
        model: grepvec::types::ModelChoice::from_str(&cli.model)?,
        bit_width: cli.bits,
        chunk_size: cli.chunk_size,
        top_k: cli.top_k,
        no_cache: cli.no_cache,
        json_output: use_json,
        files_only: cli.files_only,
        reindex: cli.reindex,
        ignore_files: cli.ignore_file.clone(),
    };

    let results = if cli.stdin {
        use std::io::Read;
        let mut stdin_text = String::new();
        std::io::stdin().read_to_string(&mut stdin_text)?;
        grepvec::stdin_search(query, &stdin_text, &config)?
    } else {
        let paths = if cli.paths.is_empty() { vec![PathBuf::from(".")] } else { cli.paths.clone() };
        grepvec::implicit_search(query, &paths, &config)?
    };

    if results.is_empty() {
        if use_json { println!("[]"); }
        process::exit(1);
    }
    print_results(&results, &config, cli.fields.as_deref())?;
    Ok(())
}

fn print_results(results: &[grepvec::types::SearchResult], config: &grepvec::types::Config, fields: Option<&str>) -> Result<()> {
    if config.json_output {
        if let Some(field_str) = fields {
            let field_list: Vec<&str> = field_str.split(',').map(str::trim).collect();
            let filtered: Vec<serde_json::Value> = results.iter().map(|r| {
                let full = serde_json::to_value(r).unwrap_or_default();
                let mut obj = serde_json::Map::new();
                for f in &field_list {
                    if let Some(v) = full.get(*f) { obj.insert((*f).to_string(), v.clone()); }
                }
                serde_json::Value::Object(obj)
            }).collect();
            println!("{}", serde_json::to_string_pretty(&filtered)?);
        } else {
            println!("{}", serde_json::to_string_pretty(results)?);
        }
    } else if config.files_only {
        let mut seen = std::collections::HashSet::new();
        for r in results {
            if seen.insert(&r.path) { println!("{}", r.path.display()); }
        }
    } else {
        for r in results {
            let snippet = r.chunk_text.lines().next().unwrap_or_default();
            let snippet = if snippet.len() > 120 { &snippet[..120] } else { snippet };
            println!("{}:{}  {:.3}  {}", r.path.display(), r.byte_offset, r.score, snippet);
        }
    }
    Ok(())
}
