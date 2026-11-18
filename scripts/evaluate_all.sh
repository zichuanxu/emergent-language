#!/usr/bin/env bash
# Evaluate every finished run under runs/<group>/ (CPU, exact categorical utterances).
#   bash scripts/evaluate_all.sh main [extra evaluate args]
# A verbal run runs/<group>/<world>_comm*/sK is paired with the no-communication baseline
# runs/main/<world>_nocomm/sK (the seed index wraps if there are fewer baselines).
# FORCE=1 re-evaluates runs that already have an eval.json. PYTHON selects the interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
group=${1:?group}; shift || true
py=${PYTHON:-python}
for final in $(find "runs/$group" -name final.pt | sort); do
    dir=$(dirname "$final")
    [[ -e "$dir/eval.json" && -z "${FORCE:-}" ]] && continue
    world=$($py -c "import json,sys; print(json.load(open(sys.argv[1]))['train']['case'])" "$dir/config.json")
    comm=$($py -c "import json,sys; c=json.load(open(sys.argv[1]))['train']; print(int(c['comm'] and c['env']['comm']))" "$dir/config.json")
    args=()
    if [[ "$comm" == 1 ]]; then
        bases=( $(ls -d runs/main/${world}_nocomm/s* 2>/dev/null || true) )
        if [[ ${#bases[@]} -gt 0 ]]; then
            seed=$($py -c "import json,sys; print(json.load(open(sys.argv[1]))['train']['seed'])" "$dir/config.json")
            args=(--baseline "${bases[$(( (seed - 1) % ${#bases[@]} ))]}")
        fi
    fi
    echo "== $dir ${args[*]:-}"
    # ${args[@]+...}: bash 3.2 (macOS) treats an empty array as unset under `set -u`
    $py -m emlang.evaluate "$dir" ${args[@]+"${args[@]}"} "$@"
done
