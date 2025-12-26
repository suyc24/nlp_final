#!/usr/bin/env bash
set -euo pipefail

# Usage: run from anywhere; script finds repo-relative paths.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="$REPO_ROOT/config.py"
EVAL_SCRIPT="$REPO_ROOT/test/eval_gsm8k.py"
LOG_DIR="$REPO_ROOT/logs/eval_threshold_sweep_7"

mkdir -p "$LOG_DIR"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
  echo "Eval script not found: $EVAL_SCRIPT" >&2
  exit 1
fi

# backup
cp "$CONFIG_FILE" "$CONFIG_FILE.bak"

# thresholds to try (adjust as needed)
thresholds=(0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4)

for thr in "${thresholds[@]}"; do
  echo "================================================================"
  echo "Running with SIMILARITY_THRESHOLD=$thr"
  # replace the line in config.py
  sed -i -E "s/^SIMILARITY_THRESHOLD = .*$/SIMILARITY_THRESHOLD = $thr/" "$CONFIG_FILE"

  OUTFILE="$LOG_DIR/eval_threshold_${thr//./_}.log"
  echo "Log -> $OUTFILE"

  # run the eval (cd into test to preserve relative imports) and capture output
  (cd "$REPO_ROOT/test" && python3 eval_gsm8k.py) 2>&1 | tee "$OUTFILE"
done

# restore original config
mv "$CONFIG_FILE.bak" "$CONFIG_FILE"

echo "Sweep completed. Logs saved in: $LOG_DIR"
