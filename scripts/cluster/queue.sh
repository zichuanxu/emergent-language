#!/usr/bin/env bash
# Run a list of jobs with at most PAR containers at a time (run on the GPU server, from the project directory).
#   GPU=0 PAR=6 bash src/scripts/cluster/queue.sh src/experiments/suite_main.txt
# Each line of the list is:  RUN MODULE ARGS...   (blank lines and # comments are skipped)
# The same command resumes an interrupted queue: an evaluation whose runs/RUN/eval.json exists is
# skipped, and a training job that already reached its last update exits within seconds from its
# checkpoint. With GPU set, a job waits until the card's used memory is below MAXUSED MiB (it is
# never skipped). If a container exits with 75 (someone stopped it to free the card) the queue
# stops instead of taking the card again. Environment: see job.sh.
set -uo pipefail
R=${EMLANG_REMOTE:-$PWD}
PROJECT=${EMLANG_PROJECT:-emlang}
here=src/scripts/cluster
list=${1:?job list}
par=${PAR:-6}
cd "$R"
mine=(--filter "label=project=$PROJECT" --filter "label=owner=$(id -un)")
running() { docker ps -q "${mine[@]}" | wc -l; }
while read -r run module args; do
    [[ -z "$run" || "$run" == \#* ]] && continue
    if [[ -e "runs/$run/eval.json" && "$module" == emlang.evaluate ]]; then
        echo "skip $run"; continue
    fi
    while [[ $(running) -ge $par ]]; do sleep 5; done
    if [[ -n "${GPU:-}" ]]; then
        while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU") -ge ${MAXUSED:-30000} ]]; do sleep 30; done
    fi
    if docker ps -a "${mine[@]}" --filter status=exited --format '{{.Status}}' | grep -q 'Exited (75)'; then
        echo "a job was stopped on request (exit 75): queue halted"; bash $here/reap.sh; exit 75
    fi
    bash $here/reap.sh > /dev/null
    while true; do
        # shellcheck disable=SC2086
        MAXUSED=${MAXUSED:-30000} bash $here/job.sh "$run" "$module" $args; code=$?
        [[ $code -eq 0 ]] && break
        # 3: another job claimed the memory between the check above and the launch; anything else is an error
        [[ $code -eq 3 ]] || { echo "could not start $run (exit $code): queue halted"; exit "$code"; }
        sleep 60
    done
    [[ -n "${GPU:-}" ]] && sleep "${SETTLE:-90}"      # let the new job claim its memory before the next check
done < "$list"
while [[ $(running) -gt 0 ]]; do sleep 5; done
bash $here/reap.sh
echo "queue finished"
