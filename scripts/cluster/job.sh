#!/usr/bin/env bash
# Start one detached training or evaluation container (run on the GPU server, from the project directory).
#   bash src/scripts/cluster/job.sh RUN MODULE [module arguments ...]
#   GPU=0 bash src/scripts/cluster/job.sh RUN emlang.train --case 1x1x3 ...
#
# Environment:
#   EMLANG_IMAGE    Docker image with Python and PyTorch (required)
#   EMLANG_REMOTE   project directory on the server (default: the current directory)
#   EMLANG_PREFIX   container name prefix (default: emlang)
#   EMLANG_PROJECT  value of the container's project label (default: emlang)
#   GPU             GPU index; without it the container sees no GPU
#   MAXUSED         refuse to start if the GPU already has this many MiB in use (default 500)
#   CPUS, MEM       container limits (defaults 4 and 12g)
#   REFRESH=1       replace the run's source snapshot by the current src/ (the old one is kept, renamed)
#
# Every container carries owner/project/stop labels, CPU/memory/pid limits, no network, and a
# read-only snapshot of the source taken at its first launch. The job checkpoints on SIGTERM and
# exits 75; rerunning the same command resumes it. Finished containers are archived and removed
# by reap.sh.
set -euo pipefail
R=${EMLANG_REMOTE:-$PWD}
PROJECT=${EMLANG_PROJECT:-emlang}
IMAGE=${EMLANG_IMAGE:?set EMLANG_IMAGE to a Docker image with PyTorch}
run=${1:?run name}; module=${2:?python module}; shift 2
[[ "$run" =~ ^[a-zA-Z0-9_.-]+$ ]]
cpus=${CPUS:-4}
name="${EMLANG_PREFIX:-emlang}_$run"
snapshot="$R/configs/$run.source"
mkdir -p "$R/runs/$run" "$R/logs" "$R/configs"
if [[ -n "${REFRESH:-}" && -e "$snapshot" ]]; then      # resume with corrected source: keep the old snapshot
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    mv "$snapshot" "$snapshot.$stamp"; mv "$R/configs/$run.sha256" "$R/configs/$run.sha256.$stamp"
fi
if [[ ! -e "$snapshot" ]]; then
    mkdir "$snapshot"
    cp -R "$R/src/." "$snapshot/"
    (cd "$snapshot" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$R/configs/$run.sha256"
    docker image inspect --format '{{.Id}}' "$IMAGE" > "$R/configs/$run.image"
fi
printf '%s utc=%s gpu=%s cpus=%s args: %s %s\n' "$name" "$(date -u +%FT%TZ)" "${GPU:-none}" "$cpus" "$module" "$*" >> "$R/configs/$run.attempts"
image=$(cat "$R/configs/$run.image")
if [[ -n "${GPU:-}" ]]; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [[ "$used" -lt "${MAXUSED:-500}" ]] || { echo "GPU $GPU is in use (${used} MiB). Not starting."; exit 3; }
    device=(--gpus "device=$GPU")
else
    device=(-e NVIDIA_VISIBLE_DEVICES=void)
fi
docker run -d --name "$name" --network none "${device[@]}" \
    --user "$(id -u):$(id -g)" --cpus "$cpus" --memory "${MEM:-12g}" --memory-swap "${MEM:-12g}" --pids-limit 512 \
    --label "owner=$(id -un)" --label "project=$PROJECT" --label "run=$run" --label stoppable=true \
    --label "stop-cmd=docker stop -t 120 $name" \
    --label "resume-cmd=cd $R && bash src/scripts/cluster/reap.sh && ${GPU:+GPU=$GPU }bash src/scripts/cluster/job.sh $run $module $*" \
    --stop-signal SIGTERM --stop-timeout 120 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH=/proj/scripts -e OMP_NUM_THREADS="$cpus" -e MKL_NUM_THREADS="$cpus" \
    --mount "type=bind,src=$snapshot,dst=/proj/scripts,readonly" \
    --mount "type=bind,src=$R/runs,dst=/proj/runs,readonly" \
    --mount "type=bind,src=$R/runs/$run,dst=/proj/results" \
    -w /proj/scripts --entrypoint python "$image" -u -m "$module" "$@" >/dev/null
echo "started $name"
