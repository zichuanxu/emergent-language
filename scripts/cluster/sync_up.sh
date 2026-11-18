#!/usr/bin/env bash
# Laptop -> server: copy the source tree into <remote>/src/ (nothing is deleted on the server).
#   EMLANG_HOST=user@server EMLANG_REMOTE=/path/on/server bash scripts/cluster/sync_up.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
host=${EMLANG_HOST:?set EMLANG_HOST to the ssh host of the GPU server}
remote=${EMLANG_REMOTE:?set EMLANG_REMOTE to the project directory on the server}
ssh -o BatchMode=yes "$host" "mkdir -p '$remote'/{src,configs,runs,logs}"
rsync -a --exclude __pycache__ emlang scripts experiments "$host:$remote/src/"
