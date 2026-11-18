#!/usr/bin/env bash
# Server -> laptop: copy finished or in-progress runs into runs/<group>/.
#   EMLANG_HOST=user@server EMLANG_REMOTE=/path/on/server bash scripts/cluster/sync_down.sh main '1x1x3_*' 'guiding'
# Patterns are matched against run names on the server.
set -euo pipefail
cd "$(dirname "$0")/../.."
host=${EMLANG_HOST:?set EMLANG_HOST to the ssh host of the GPU server}
remote=${EMLANG_REMOTE:?set EMLANG_REMOTE to the project directory on the server}
group=${1:?local group directory under runs/}; shift
mkdir -p "runs/$group"
for pattern in "$@"; do
    rsync -a --include="$pattern/" --include="$pattern/**" --exclude='*' "$host:$remote/runs/" "runs/$group/"
done
