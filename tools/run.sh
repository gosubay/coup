#!/usr/bin/env bash
# Drive the Coup solve in stages, smallest tree first.
#
#   tools/run.sh start     launch the whole plan detached, survives logout
#   tools/run.sh status    where each stage is, and whether it has flattened
#   tools/run.sh stop      halt after the current stage checkpoints
#
# Every stage checkpoints and resumes, so stopping and restarting loses nothing.
# Iteration counts are a ceiling, not a target: the answer is done when NashConv
# stops falling, which is what `status` is for.

set -u
cd "$(dirname "$0")/.." || exit 1

OUT=${OUT:-solves}
# leave the machine a third of its RAM; a solve that hits this checkpoints and
# stops cleanly instead of being OOM-killed
MAXGB=${MAXGB:-$(awk '/MemTotal/ {printf "%.1f", $2/1048576*0.66}' /proc/meminfo)}
STAGES=(
  # name          peek claim  iters
  "base            0    0     20000000"
  "claims          0    1     60000000"
  "full            1    1    150000000"
)

# override for a quick trial:  COUP_STAGES="trial 0 0 200000" tools/run.sh start
if [ -n "${COUP_STAGES:-}" ]; then
  IFS=';' read -r -a STAGES <<<"$COUP_STAGES"
fi

launch() {
  mkdir -p "$OUT"
  rm -f "$OUT/STOP"
  echo "plan:"
  for st in "${STAGES[@]}"; do
    set -- $st; printf "  %-8s peek=%s claim=%s  up to %s iters\n" "$1" "$2" "$3" "$4"
  done
  echo
  echo "memory budget ${MAXGB} GB -- a stage that reaches it checkpoints and stops"
  echo "logs in $OUT/, checkpoints in $OUT/*.pkl"
  echo "watch with: tools/run.sh status"
  nohup bash "$0" _work >"$OUT/driver.log" 2>&1 &
  echo "driver pid $!"
}

work() {
  mkdir -p "$OUT"
  python3 tools/solve_coup.py verify 2>&1 | tee "$OUT/verify.log"
  echo
  for st in "${STAGES[@]}"; do
    set -- $st; name=$1 peek=$2 claim=$3 iters=$4
    [ -f "$OUT/STOP" ] && { echo "stopping: STOP file present"; return; }
    pkl="$OUT/$name.pkl"
    echo "=== stage $name (peek=$peek claim=$claim, up to $iters iters) $(date -u +%H:%M) ==="
    python3 tools/solve_coup.py solve \
      --peek-memory "$peek" --claim-memory "$claim" \
      --iters "$iters" --out "$pkl" --resume --max-gb "$MAXGB" \
      --report-every 500000 --checkpoint-every 2000000 \
      --exploit-every 5000000 >>"$OUT/$name.log" 2>&1
    python3 tools/solve_coup.py exploit --solve "$pkl" >>"$OUT/$name.log" 2>&1
    # only the opening-action table: a full-phase dump of the big configurations
    # runs to tens of GB. Use `query` for everything else.
    python3 tools/solve_coup.py export --solve "$pkl" --csv "$OUT/$name-actions.csv" \
      --phase action --min-prob 0.005 >>"$OUT/$name.log" 2>&1
    echo "=== stage $name done $(date -u +%H:%M) ==="
  done
  echo "ALL STAGES DONE"
}

status() {
  for st in "${STAGES[@]}"; do
    set -- $st; name=$1
    [ -f "$OUT/$name.pkl" ] || { printf "%-8s not started\n" "$name"; continue; }
    echo "--- $name"
    python3 tools/solve_coup.py stats --solve "$OUT/$name.pkl" | sed 's/^/  /'
    echo
  done
  pgrep -f "solve_coup.py solve" >/dev/null && echo "solver is running" || echo "solver is NOT running"
}

case "${1:-status}" in
  start)  launch ;;
  _work)  work ;;
  status) status ;;
  stop)   mkdir -p "$OUT"; touch "$OUT/STOP"
          pkill -f "solve_coup.py solve"
          echo "stopped. checkpoints kept; 'start' resumes where it left off" ;;
  *)      sed -n '2,12p' "$0"; exit 1 ;;
esac
