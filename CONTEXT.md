# grepvec -- Agent Context

## What it does

`grepvec` is a hybrid vector + fulltext search CLI. Point it at a directory, give it a query, get ranked results. No index build step, no daemons.

## Rules for agents

- Always use `--json` for all commands to get machine-readable output.
- Always use `--dry-run` before executing mutating operations (`add`, `rm`, `gc --aggressive`, `init`).
- Always use `--fields` to limit search response size (e.g., `--fields "path,score"`).
- Use `grepvec schema <command>` to discover exact parameters before calling a command.
- Use `--no-embed` for text-only (BM25) search when semantic similarity isn't needed -- avoids model download.
- Use `--model bge-small` for faster embedding on small corpora (~30 MB vs ~400 MB download).
- Output is auto-JSON when stdout is piped (non-TTY). Errors are JSON on stderr when piped.
- Exit codes: 0 = matches found, 1 = no matches, 2 = error.
- The first run downloads an embedding model (~400 MB for default `bge-large`). Use `--model bge-small` (~30 MB) to avoid large downloads.

## Common patterns

```bash
# Search with minimal output
grepvec --json --fields "path,score" "my query" ./src

# BM25-only search (no model download)
grepvec --json --no-embed "exact phrase" ./docs

# Check what would be added before adding
grepvec add ./corpus ./new-files --dry-run

# Get command schema
grepvec schema search
```

## Do NOT

- Do not call mutating commands without `--dry-run` first.
- Do not omit `--json` -- human-formatted output is not parseable.
- Do not use `--top` values above 50 -- large result sets waste context.
- Do not use `grepvec gc --aggressive` without confirming with the user.
