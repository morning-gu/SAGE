#!/usr/bin/env bash
# ============================================================================
# SAGE test wrapper
# ----------------------------------------------------------------------------
# Thin wrapper around pytest tests/.  Exits 0 with a hint when tests/ does not
# exist (the repo currently keeps its validation in scripts and dry-runs).
#
# Historical note: run_tests.sh used to be a renamed copy of the latency-eval
# runner whose usage block referenced scripts/run_eval.sh (a file that never
# existed at that path).  That legacy eval-runner logic has been superseded by
# scripts/run_experiment.py (see configs/experiments/) and lives only in git
# history (commit 2ff488b "rename: move run_latency_tests.sh to run_tests.sh").
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"   # repo root

if [ ! -d tests ]; then
  echo "[run_tests] no tests/ directory - nothing to run"
  echo "[run_tests] hint: experiment validation lives in scripts/run_experiment.py"
  echo "             (python scripts/run_experiment.py --list, --dry-run)"
  exit 0
fi

exec python -m pytest tests/ "$@"
