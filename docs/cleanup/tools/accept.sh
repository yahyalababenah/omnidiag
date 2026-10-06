#!/bin/bash
# Gate acceptance, from the repo root, on the STAGED change set.
# usage: docs/cleanup/tools/accept.sh <work-dir> [pytest-marker-expr|full] [--build-trace]
#   1. pytest in the normal tree (marker expr, or the full suite with "full")
#   2. crosscheck.py in the normal tree, diffed with baselines/crosscheck_B4.txt
#   3. the same two in a clean worktree (HEAD + staged diff; heart weights linked, no BRFSS files)
#   4. with --build-trace: both Docker build steps traced in the clean worktree, every read
#      checked against .dockerignore, plus a static check that the Dockerfile names no models/diabetes
set -u
ROOT=$(pwd); WORK=$1; MARK=${2:-not brfss}; TRACE=${3:-}
PY=$ROOT/backend/.venv/bin/python; T=$ROOT/docs/cleanup/tools; B=$ROOT/docs/cleanup/baselines
mkdir -p "$WORK"; rm -f "$WORK"/*.db
sel=(); [ "$MARK" != "full" ] && sel=(-m "$MARK")
run_tests() { timeout 1500 "$PY" -m pytest -q -p no:cacheprovider --no-header -W ignore "${sel[@]}" 2>&1 | grep -E "^(FAILED|ERROR) |passed|failed" | sed -E 's/ - .*//' | tail -8; }
run_cross() { rm -f "$WORK/x.db"; DATABASE_URL="sqlite+aiosqlite:///$WORK/x.db" PYTHONHASHSEED=0 timeout 900 "$PY" "$T/crosscheck.py" > "$WORK/x_$1.txt" 2>&1
  if diff <(grep -E "^(OK|FAIL|DIGEST)" "$B/crosscheck_B4.txt") <(grep -E "^(OK|FAIL|DIGEST)" "$WORK/x_$1.txt") > "$WORK/x_$1.diff"; then echo "crosscheck ($1): IDENTICAL to baseline"; else echo "crosscheck ($1): DIFFERS"; cat "$WORK/x_$1.diff"; fi; }
echo "== normal tree, pytest ${MARK}"; run_tests; run_cross normal
W=$WORK/wt; git worktree remove --force "$W" 2>/dev/null; rm -rf "$W"
git worktree add -q --detach "$W" HEAD 2>&1 | grep -v -i lfs
git diff --cached --binary | (cd "$W" && git apply --index) || { echo "PATCH FAILED"; exit 1; }
for f in $(git ls-files --others --ignored --exclude-standard models/heart_disease data | grep -v -E "diabetes/|__pycache__|\.bak"); do [ -e "$W/$f" ] || { mkdir -p "$W/$(dirname "$f")"; ln -s "$ROOT/$f" "$W/$f"; }; done
echo "== clean worktree (no BRFSS files: $(ls "$W"/models/diabetes/*.pkl 2>/dev/null | wc -l) pkl), pytest ${MARK}"
(cd "$W" && run_tests && run_cross worktree)
if [ "$TRACE" = "--build-trace" ]; then
  echo "== build trace (clean worktree, fresh heart bundle)"
  rm -f "$WORK"/trace_*.txt
  (cd "$W" && rm -f models/heart_disease/heart_l3_glm_stack.pkl && mkdir -p models/heart_disease \
    && TRACE_OUT="$WORK/trace_heart.txt" PYTHONDONTWRITEBYTECODE=1 "$PY" "$T/trace.py" scripts/train_heart_glm.py --verify > "$WORK/b1.log" 2>&1; echo "train_heart_glm --verify exit=$?" \
    && TRACE_OUT="$WORK/trace_drift.txt" PYTHONDONTWRITEBYTECODE=1 "$PY" "$T/trace.py" scripts/build_drift_reference.py --verify > "$WORK/b2.log" 2>&1; echo "build_drift_reference --verify exit=$?")
  # a cached import reads __pycache__/x.cpython-NN.pyc, not x.py: map it back to its source
  cat "$WORK"/trace_*.txt | awk -F'\t' '$2 !~ /w/ {print $1}' | sed -E 's#__pycache__/([^/]+)\.cpython-[0-9]+\.pyc$#\1.py#' | sort -u > "$WORK/reads.txt"
  echo "build-time reads: $(wc -l < "$WORK/reads.txt") files"; (cd "$W" && "$PY" "$T/dockerignore_check.py" .dockerignore < "$WORK/reads.txt" | grep -E "self-test|EXCLUDED"; echo "dockerignore check exit=${PIPESTATUS[0]}")
  grep -n "models/diabetes/" "$W/Dockerfile" && echo "Dockerfile still names models/diabetes/" || echo "Dockerfile names no models/diabetes/"
  grep -n -i -E "models/diabetes/|data/diabetes/|configs/diabetes\.yaml|brfss" "$WORK/reads.txt" || echo "no BRFSS file read at build time"
fi
find "$W" -type l -delete; git worktree remove --force "$W"; git worktree prune
