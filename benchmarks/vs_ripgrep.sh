#!/usr/bin/env bash
#
# grepvec vs ripgrep benchmark
#
# Tests two things:
#   1. Speed: ripgrep (text) vs grepvec (semantic, cached) vs grepvec (cold embed)
#   2. Quality: queries where ripgrep returns 0 results but grepvec finds relevant docs
#
# Usage: ./benchmarks/vs_ripgrep.sh [--model bge-small]
#
set -uo pipefail

MODEL="bge-small"
if [ "${1:-}" = "--model" ] && [ -n "${2:-}" ]; then
    MODEL="$2"
fi
CORPUS_DIR=$(mktemp -d)
CACHE_DIR=$(mktemp -d)
# Build once, then use the binary directly to avoid cargo overhead.
cargo build --quiet 2>/dev/null
GREPVEC="./target/debug/grepvec"
RG="rg"
NUM_FILES=50

echo "=== grepvec vs ripgrep benchmark ==="
echo "Model: $MODEL | Corpus: $NUM_FILES files | Cache: $CACHE_DIR"
echo ""

# ---------------------------------------------------------------------------
# Generate a realistic corpus of technical notes
# ---------------------------------------------------------------------------
echo "Generating $NUM_FILES test files..."

cat > "$CORPUS_DIR/001_rate_limiting.md" << 'EOF'
# Rate Limiting at the Gateway

The gateway needs to shed load when downstream services are slow.
Token bucket with burst support is the right algorithm here.
We should set the refill rate based on p99 latency of the backend.
EOF

cat > "$CORPUS_DIR/002_circuit_breakers.md" << 'EOF'
# Circuit Breaker Pattern

When a dependency fails repeatedly, stop calling it.
The half-open state probes with a single request to check recovery.
Use exponential backoff for retry intervals.
EOF

cat > "$CORPUS_DIR/003_event_pipeline.md" << 'EOF'
# Event Pipeline Architecture

Kafka for ordered event streaming. Dead letter queues for poison messages.
Each consumer group processes events independently.
Exactly-once semantics require idempotent consumers.
EOF

cat > "$CORPUS_DIR/004_database_scaling.md" << 'EOF'
# Scaling the Database Layer

Read replicas handle query traffic. Write primary handles mutations.
Connection pooling with PgBouncer reduces overhead.
Consider sharding by tenant ID when single-node hits limits.
EOF

cat > "$CORPUS_DIR/005_auth_system.md" << 'EOF'
# Authentication and Authorization

JWT tokens for stateless auth. Refresh tokens stored server-side.
RBAC with hierarchical roles. Permission checks at the middleware layer.
OAuth2 for third-party integrations.
EOF

cat > "$CORPUS_DIR/006_caching_strategy.md" << 'EOF'
# Caching Strategy

Redis for hot data. Cache invalidation on write-through.
TTL-based expiry for eventually consistent reads.
Cache stampede protection with probabilistic early expiry.
EOF

cat > "$CORPUS_DIR/007_deployment.md" << 'EOF'
# Deployment Pipeline

Blue-green deploys minimize downtime. Canary releases catch regressions early.
Feature flags decouple deploy from release.
Rollback within 60 seconds if error rate spikes.
EOF

cat > "$CORPUS_DIR/008_observability.md" << 'EOF'
# Observability Stack

Structured logging with trace IDs. Metrics via Prometheus.
Distributed tracing with OpenTelemetry spans.
Alert on SLO burn rate, not raw error counts.
EOF

cat > "$CORPUS_DIR/009_api_design.md" << 'EOF'
# API Design Principles

REST for CRUD, GraphQL for flexible queries.
Pagination with cursor-based tokens, not offset.
Rate limit by API key with tiered quotas.
EOF

cat > "$CORPUS_DIR/010_testing.md" << 'EOF'
# Testing Strategy

Unit tests at module boundaries. Integration tests against real databases.
Contract tests between services. Load tests before major releases.
Snapshot tests for generated output.
EOF

# Generate more files with varied content
for i in $(seq 11 $NUM_FILES); do
    cat > "$CORPUS_DIR/$(printf '%03d' $i)_filler.md" << EOF
# Technical Note $i

This document covers implementation details for component $i.
Performance considerations include memory allocation patterns and cache locality.
Error handling follows the result type pattern with structured error codes.
The module exposes a clean interface with minimal public surface area.
Configuration is loaded from environment variables at startup.
EOF
done

echo "Created $NUM_FILES files in $CORPUS_DIR"
echo ""

# ---------------------------------------------------------------------------
# Queries: exact match (ripgrep wins) vs semantic (grepvec wins)
# ---------------------------------------------------------------------------

# Exact queries that ripgrep handles well
EXACT_QUERIES=(
    "token bucket"
    "circuit breaker"
    "Kafka"
    "PgBouncer"
    "JWT tokens"
)

# Semantic queries where the words DON'T appear in the text
SEMANTIC_QUERIES=(
    "how to handle too many requests"                  # -> rate limiting (no exact match)
    "what to do when a service keeps failing"          # -> circuit breakers
    "message queue ordering guarantees"                # -> event pipeline / Kafka
    "database read performance bottleneck"             # -> database scaling / read replicas
    "user login security"                              # -> auth system
    "speed up repeated data lookups"                   # -> caching strategy
    "zero downtime releases"                           # -> deployment / blue-green
    "finding the root cause of production issues"      # -> observability
    "how to paginate API results efficiently"          # -> API design / cursor-based
    "making sure code changes don't break things"      # -> testing strategy
)

# ---------------------------------------------------------------------------
# Speed benchmark: cold embed
# ---------------------------------------------------------------------------
echo "--- SPEED: Cold embed (first run, includes embedding) ---"
START=$(python3 -c 'import time; print(time.time())')
$GREPVEC --model "$MODEL" --cache-dir "$CACHE_DIR" --json --top 3 \
    "rate limiting" "$CORPUS_DIR" > /dev/null 2>&1
END=$(python3 -c 'import time; print(time.time())')
COLD_TIME=$(python3 -c "print(f'{$END - $START:.2f}')")
echo "  grepvec cold:  ${COLD_TIME}s (includes model load + embedding $NUM_FILES files)"
echo ""

# ---------------------------------------------------------------------------
# Speed benchmark: warm (cached)
# ---------------------------------------------------------------------------
echo "--- SPEED: Warm queries (cached embeddings + .tv index) ---"
echo ""
printf "  %-50s %10s %10s\n" "Query" "ripgrep" "grepvec"
printf "  %-50s %10s %10s\n" "-----" "-------" "-------"

for q in "${EXACT_QUERIES[@]}"; do
    # ripgrep
    RG_START=$(python3 -c 'import time; print(time.time())')
    $RG -l "$q" "$CORPUS_DIR" > /dev/null 2>&1 || true
    RG_END=$(python3 -c 'import time; print(time.time())')
    RG_MS=$(python3 -c "print(f'{($RG_END - $RG_START)*1000:.0f}')")

    # grepvec (cached)
    GV_START=$(python3 -c 'import time; print(time.time())')
    $GREPVEC --model "$MODEL" --cache-dir "$CACHE_DIR" --json --top 3 \
        "$q" "$CORPUS_DIR" > /dev/null 2>&1
    GV_END=$(python3 -c 'import time; print(time.time())')
    GV_MS=$(python3 -c "print(f'{($GV_END - $GV_START)*1000:.0f}')")

    printf "  %-50s %8sms %8sms\n" "\"$q\"" "$RG_MS" "$GV_MS"
done
echo ""

# ---------------------------------------------------------------------------
# Quality: semantic queries
# ---------------------------------------------------------------------------
echo "--- QUALITY: Semantic queries (words not in text) ---"
echo ""

SEMANTIC_HITS=0
SEMANTIC_TOTAL=${#SEMANTIC_QUERIES[@]}

for q in "${SEMANTIC_QUERIES[@]}"; do
    # ripgrep: try to find something
    RG_COUNT=$($RG -c "$q" "$CORPUS_DIR" 2>/dev/null | wc -l | tr -d ' ')

    # grepvec: semantic search
    GV_TOP=$($GREPVEC --model "$MODEL" --cache-dir "$CACHE_DIR" --json --top 1 \
        "$q" "$CORPUS_DIR" 2>/dev/null || echo "[]")
    GV_PATH=$(echo "$GV_TOP" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    if r: print(r[0].get('path','').split('/')[-1])
    else: print('(none)')
except: print('(none)')
" 2>/dev/null)
    GV_SCORE=$(echo "$GV_TOP" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    if r: print(f'{r[0].get(\"score\",0):.3f}')
    else: print('0.000')
except: print('0.000')
" 2>/dev/null)

    if [ "$RG_COUNT" = "0" ] && [ "$GV_PATH" != "(none)" ]; then
        SEMANTIC_HITS=$((SEMANTIC_HITS + 1))
        STATUS="GREPVEC WINS"
    elif [ "$RG_COUNT" != "0" ]; then
        STATUS="both found"
    else
        STATUS="neither"
    fi

    printf "  %-50s rg:%-3s gv:%-25s %s\n" \
        "\"$(echo $q | head -c 47)\"" "$RG_COUNT" "$GV_PATH($GV_SCORE)" "$STATUS"
done

echo ""
echo "--- SUMMARY ---"
echo "  Semantic queries where grepvec found results and ripgrep didn't: $SEMANTIC_HITS/$SEMANTIC_TOTAL"
echo "  Cold embed time ($NUM_FILES files): ${COLD_TIME}s"
echo ""

# Cleanup
rm -rf "$CORPUS_DIR" "$CACHE_DIR"
echo "Done."
