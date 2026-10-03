#!/usr/bin/env bash
# Keep run_model_passes.py going until every pass is finished.
#
# Restarts it when it exits with an error, and kills and restarts it when it
# stalls (neither the log nor progress.json touched for STALL_SECONDS). Every
# restart resumes from the per-model CSV checkpoints, so nothing is redone.
# After a reboot or power cut, start this again the same way:
#
#     nohup ./run_smells.sh > /dev/null 2>&1 &
#     tail -f runs/smells_6000/run.log
#     .venv/bin/python run_model_passes.py status
#
# Extra arguments are passed through, e.g. ./run_smells.sh --models tev1,kev

set -u
cd "$(dirname "$0")"

RUN_DIR=${RUN_DIR:-runs/smells_6000}
LIMIT=${LIMIT:-6000}
STALL_SECONDS=${STALL_SECONDS:-1200}
MAX_RESTARTS=${MAX_RESTARTS:-30}

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/run.log"
echo $$ > "$RUN_DIR/supervisor.pid"

say() { echo "$(date '+%Y-%m-%d %H:%M:%S') supervisor: $*" >> "$LOG"; }

child=
stop() {
    say "stopping"
    [ -n "$child" ] && kill -TERM "$child" 2>/dev/null && wait "$child"
    rm -f "$RUN_DIR/supervisor.pid"
    exit 130
}
trap stop INT TERM

restarts=0
while :; do
    .venv/bin/python run_model_passes.py run --run-dir "$RUN_DIR" --limit "$LIMIT" "$@" >> "$LOG" 2>&1 &
    child=$!
    say "started run_model_passes.py (pid $child)"

    while kill -0 "$child" 2>/dev/null; do
        sleep 30 & wait $!
        newest=$(stat -c %Y "$LOG" "$RUN_DIR/progress.json" 2>/dev/null | sort -n | tail -1)
        if (( $(date +%s) - ${newest:-0} > STALL_SECONDS )); then
            say "no progress for ${STALL_SECONDS}s - killing pid $child"
            kill -TERM "$child" 2>/dev/null
            sleep 20
            kill -KILL "$child" 2>/dev/null
        fi
    done
    wait "$child"
    code=$?
    child=

    if [ "$code" -eq 0 ]; then
        say "all passes finished"
        rm -f "$RUN_DIR/supervisor.pid"
        exit 0
    fi
    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
        say "exit code $code; giving up after $MAX_RESTARTS restarts"
        rm -f "$RUN_DIR/supervisor.pid"
        exit 1
    fi
    say "exit code $code; restart $restarts/$MAX_RESTARTS in 30s"
    sleep 30 & wait $!
done
