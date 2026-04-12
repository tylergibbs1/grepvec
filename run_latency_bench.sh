#!/usr/bin/env bash
# Measure search latency while maintaining recall.
# Primary metric: microseconds per query (lower is better).
# Recall is enforced by assertions in the benchmark — if they fail, this crashes.
set -euo pipefail

OUTPUT=$(cargo test --test recall_benchmark recall_turboquant_dim384 -- --ignored --nocapture 2>&1)

# If recall assertions fail, report as crash
if ! echo "$OUTPUT" | grep -q "test result: ok"; then
    echo "recall_check: FAIL"
    echo "$OUTPUT" | grep -E "(FAILED|panicked|assert)" | head -5
    exit 1
fi

# Extract recall values for secondary metrics
R1_4BIT=$(echo "$OUTPUT" | grep -A7 "4-bit TurboQuant" | grep "k= 1" | head -1 | awk '{print $NF}')
R64_2BIT=$(echo "$OUTPUT" | grep -A7 "2-bit TurboQuant" | grep "k=64" | head -1 | awk '{print $NF}')

# Run latency benchmark: 5000 vectors, dim=384, 100 queries, k=10
# This is the common use case — search for top-10 results.
LATENCY_OUTPUT=$(cargo test --test latency_benchmark -- --ignored --nocapture 2>&1)

if ! echo "$LATENCY_OUTPUT" | grep -q "test result: ok"; then
    echo "latency_check: FAIL"
    echo "$LATENCY_OUTPUT" | tail -10
    exit 1
fi

US_PER_QUERY=$(echo "$LATENCY_OUTPUT" | grep "^us_per_query:" | awk '{print $2}')

echo "us_per_query: ${US_PER_QUERY:-999999}"
echo "recall_4bit_k1: ${R1_4BIT:-0.0000}"
echo "recall_2bit_k64: ${R64_2BIT:-0.0000}"
