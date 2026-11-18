#!/usr/bin/env bash
# Archive the logs and exit state of this project's exited containers, then remove them
# (run on the GPU server, from the project directory). Environment: see job.sh.
set -euo pipefail
R=${EMLANG_REMOTE:-$PWD}
PROJECT=${EMLANG_PROJECT:-emlang}
docker ps -a --filter "label=project=$PROJECT" --filter "label=owner=$(id -un)" \
    --filter status=exited --format '{{.Names}} {{.Label "run"}}' | while read -r name run; do
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    docker logs --timestamps "$name" > "$R/logs/$run.$stamp.stdout.log" 2> "$R/logs/$run.$stamp.stderr.log"
    docker inspect --format '{{json .State}}' "$name" > "$R/logs/$run.$stamp.state.json"
    code=$(docker inspect --format '{{.State.ExitCode}}' "$name")
    cid=$(docker inspect --format '{{.Id}}' "$name")
    docker container rm "$cid" >/dev/null
    echo "reaped $name exit=$code"
done
