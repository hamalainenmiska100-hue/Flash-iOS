#!/bin/bash
# run.sh — Autoresearch experiment runner with 5-minute timeout.
# Usage: ./run.sh bench.py --model MODEL --tokens N [extra args...]
# Usage: ./run.sh stream_infer.py --model MODEL --tokens N [extra args...]
set -euo pipefail

TIMEOUT=300  # 5 minutes
SCRIPT="${1:?Usage: ./run.sh SCRIPT --model MODEL [extra_args...]}"
shift

RESULTS_FILE="results.tsv"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "none")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/run_${TIMESTAMP}.log"

mkdir -p logs

# Initialize results.tsv if missing
if [ ! -f "$RESULTS_FILE" ]; then
    printf "commit\tmodel\tparams_B\tactive_B\ttok_sec\tttft_ms\tmem_gb\tstatus\tdescription\n" > "$RESULTS_FILE"
fi

echo "=== Experiment: uv run $SCRIPT $@ ==="
echo "Timeout: ${TIMEOUT}s | Log: $LOG | Commit: $COMMIT"
echo ""

# Run with timeout, tee to both stdout and log
if timeout "$TIMEOUT" uv run "$SCRIPT" "$@" 2>&1 | tee "$LOG"; then
    # Parse RESULT line
    RESULT_LINE=$(grep '^RESULT ' "$LOG" | tail -1 || echo "")
    if [ -n "$RESULT_LINE" ]; then
        MODEL=$(echo "$RESULT_LINE" | sed -n 's/.*model=\([^ ]*\).*/\1/p')
        PARAMS=$(echo "$RESULT_LINE" | sed -n 's/.*params_B=\([^ ]*\).*/\1/p')
        ACTIVE=$(echo "$RESULT_LINE" | sed -n 's/.*active_B=\([^ ]*\).*/\1/p')
        TPS=$(echo "$RESULT_LINE" | sed -n 's/.*tok_sec=\([^ ]*\).*/\1/p')
        TTFT=$(echo "$RESULT_LINE" | sed -n 's/.*ttft_ms=\([^ ]*\).*/\1/p')
        MEM=$(echo "$RESULT_LINE" | sed -n 's/.*mem_gb=\([^ ]*\).*/\1/p')

        # Default description from args
        DESC="$SCRIPT $*"

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\tkeep\t%s\n" \
            "$COMMIT" "$MODEL" "$PARAMS" "$ACTIVE" "$TPS" "$TTFT" "$MEM" "$DESC" >> "$RESULTS_FILE"

        echo ""
        echo "=== RESULT: ${MODEL} | ${TPS} tok/s | ${MEM} GB | KEEP ==="
    else
        echo ""
        echo "=== WARNING: No RESULT line found in output ==="
        printf "%s\tunknown\t0\t0\t0\t0\t0\tcrash\t%s (no result line)\n" \
            "$COMMIT" "$SCRIPT $*" >> "$RESULTS_FILE"
    fi
else
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -eq 124 ]; then
        echo ""
        echo "=== TIMEOUT after ${TIMEOUT}s ==="
        printf "%s\tunknown\t0\t0\t0\t0\t0\tcrash\ttimeout: %s\n" \
            "$COMMIT" "$SCRIPT $*" >> "$RESULTS_FILE"
    else
        echo ""
        echo "=== CRASHED (exit code $EXIT_CODE) ==="
        printf "%s\tunknown\t0\t0\t0\t0\t0\tcrash\tcrash(%d): %s\n" \
            "$COMMIT" "$EXIT_CODE" "$SCRIPT $*" >> "$RESULTS_FILE"
    fi
fi
