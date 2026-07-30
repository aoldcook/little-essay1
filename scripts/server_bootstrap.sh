#!/usr/bin/env bash
# Server bootstrap for little-essay1.
#
# Sets up the environment, verifies the pipeline can run, then (optionally) runs
# the data -> train -> evaluate chain.
#
# Usage:
#   bash scripts/server_bootstrap.sh setup      # env + import checks only
#   bash scripts/server_bootstrap.sh smoke      # + verify reader LLM reachable
#   bash scripts/server_bootstrap.sh data       # + build/verify dataset splits
#   bash scripts/server_bootstrap.sh train      # + train span model (group split)
#   bash scripts/server_bootstrap.sh evaluate   # + downstream EM/F1 evaluation
#   bash scripts/server_bootstrap.sh all
#
# The API key is NEVER passed as an argument and never written by this script.
# Create .env yourself (see `setup_env` output below).

set -euo pipefail

STAGE="${1:-setup}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ="$REPO_ROOT/1 Prompt Compression with Context-Aware Sentence Encoding for Fast and"
VENV="$REPO_ROOT/.venv"
PY="$VENV/bin/python"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

setup_env() {
  log "Python / GPU"
  command -v python3 >/dev/null || die "python3 not found"
  python3 --version
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no nvidia-smi (CPU-only host)"

  log "Virtualenv at $VENV"
  [ -d "$VENV" ] || python3 -m venv "$VENV"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r "$REPO_ROOT/requirements.txt"

  log "Import check"
  cd "$PROJ"
  "$PY" - <<'PYCODE'
import importlib, sys
mods = ["torch", "transformers", "numpy", "openai"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  ok {m:<14} {getattr(mod,'__version__','?')}")
    except Exception as exc:
        print(f"  FAIL {m}: {exc}"); sys.exit(1)
import torch
print(f"  cuda available: {torch.cuda.is_available()}")
PYCODE

  log "Pipeline import check"
  "$PY" -c "
import sys; sys.path.insert(0,'.')
from pipeline.compression_pipeline import ContextAwareCompressor
from pipeline.runtime_contract import resolve_encoder_source, EncoderContractError
from evaluation.qa_metrics import score_prediction
print('  pipeline imports OK')
try:
    resolve_encoder_source(None, False); print('  FAIL: contract did not raise')
except EncoderContractError: print('  encoder contract active (fails loud) OK')
"

  log "Environment manifest"
  "$PY" -m repro.manifest | head -40

  if [ ! -f "$REPO_ROOT/.env" ]; then
    cat <<'MSG'

  NOTE: no .env found. Create it now (this does not echo the key to history):

      cd "$(git rev-parse --show-toplevel)"
      install -m 600 /dev/null .env
      printf 'READER_MODEL=qwen3.6-flash\n' >> .env
      printf 'DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1\n' >> .env
      read -rsp 'DASHSCOPE_API_KEY: ' K && printf 'DASHSCOPE_API_KEY=%s\n' "$K" >> .env && unset K && echo

MSG
  fi
}

smoke() {
  log "Reader LLM smoke test (validates key + model id in seconds)"
  cd "$PROJ"
  "$PY" -m evaluation.run_downstream_eval --input_file /dev/null --smoke_test_only \
    || die "reader unreachable - check DASHSCOPE_API_KEY / READER_MODEL / DASHSCOPE_BASE_URL"
}

build_data() {
  log "Dataset splits"
  cd "$PROJ"
  local split_dir="data_builder/english_cqr_mixed_5k"
  if [ -f "$split_dir/test.jsonl" ]; then
    wc -l "$split_dir"/*.jsonl
  else
    echo "  splits absent (gitignored). Rebuild with:"
    echo "    \"$PY\" -m data_builder.build_mixed_english_cqr_training_data --help"
    echo "    \"$PY\" -m data_builder.prepare_cqr_training_splits --help"
    die "no test.jsonl - build the dataset before training/evaluating"
  fi
}

train_span() {
  log "Span model training (group-disjoint split)"
  cd "$PROJ"
  local pseudo="intra_sentence_model/span_pseudo_train.jsonl"
  [ -f "$pseudo" ] || die "missing $pseudo - generate it with intra_sentence_model.generate_span_pseudo_labels"
  "$PY" -m intra_sentence_model.train_span_model \
    --train_file "$pseudo" \
    --output_dir intra_sentence_model/outputs_english_group \
    --split_mode group --epochs 60 --seed 42
}

evaluate() {
  log "Downstream QA evaluation (real EM/F1 with frozen reader)"
  cd "$PROJ"
  local enc="${ENCODER_DIR:-}"
  local extra=()
  if [ -z "$enc" ]; then
    echo "  ENCODER_DIR not set -> running NON-NEURAL lexical baseline (labelled as such)."
    extra+=(--allow_lexical_fallback)
  else
    extra+=(--encoder_dir "$enc")
  fi
  "$PY" -m evaluation.run_downstream_eval \
    --input_file data_builder/english_cqr_mixed_5k/test.jsonl \
    --methods none,truncate,topk_lexical,ours_stage1,ours_full \
    --ratios 0.5,0.25,0.125 \
    --limit "${LIMIT:-200}" \
    --seeds "${SEEDS:-42}" \
    "${extra[@]}"
}

case "$STAGE" in
  setup)    setup_env ;;
  smoke)    setup_env; smoke ;;
  data)     setup_env; build_data ;;
  train)    setup_env; build_data; train_span ;;
  evaluate) setup_env; smoke; evaluate ;;
  all)      setup_env; smoke; build_data; train_span; evaluate ;;
  *)        die "unknown stage: $STAGE (setup|smoke|data|train|evaluate|all)" ;;
esac

log "Stage '$STAGE' complete."
