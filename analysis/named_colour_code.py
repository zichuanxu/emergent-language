"""How a trained speaker refers to the three landmarks: by one reference colour, and by silence.

    python analysis/named_colour_code.py runs/main/1x1x3_comm/s1 [more run directories ...]

For each run: the landmark colour that most of its words refer to, directly ("GREEN") or as a
reference point ("FARTHER-FROM-GREEN"), taken from each symbol's best meaning in eval.json. Then, on
the 10,000 held-out test worlds, for go-to goals addressed to another agent, split by whether the
target is that landmark, the nearer of the other two from it, or the farther one: the share of
speakers that say nothing about the landmark in the whole episode (words that clearly name an action
do not count), and the share of addressees that end nearest the target landmark. Prints a Markdown table.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emlang.evaluate import goal_factors, play, test_worlds
from emlang.train import load
from emlang.world import ACTION_NAMES, COLOR_NAMES, GOTO, _take

print("| Run | Reference colour | Target is the reference landmark: speaker says nothing about the landmark | addressee at target | "
      "Target is the nearer other landmark: says nothing | addressee at target | "
      "Target is the farther other landmark: says nothing | addressee at target |")
print("|---|---|---:|---:|---:|---:|---:|---:|")
for run in sys.argv[1:]:
    symbols = json.loads((Path(run) / "eval.json").read_text())["symbols"]
    refs = [c for row in symbols["table"].values() for c in COLOR_NAMES
            if row["label"].split("-")[-1] == c and row["factor"] != "addressee"]
    if not refs:
        print(f"| {run} | none | | | | | | |")
        continue
    colour = max(COLOR_NAMES, key=refs.count)
    action_words = [int(k) for k, label in symbols["labels"].items() if label in ACTION_NAMES]
    policy, tcfg = load(run)
    state = test_worlds(tcfg.env, 10_000)
    ep = play(policy, tcfg.env, state)
    n = state.pos.shape[1]
    directed = (state.goal_type == GOTO) & (state.goal_rcpt != torch.arange(n))
    relation = goal_factors(state)[f"relative_to_{colour}"]
    about_landmark = ep.symbols != 0
    for k in action_words:
        about_landmark &= ep.symbols != k
    silent = ~about_landmark.any(1)
    final = _take(ep.pos[:, -1], state.goal_rcpt)
    hit = (final.unsqueeze(2) - state.lm_pos.unsqueeze(1)).norm(dim=-1).argmin(-1) == state.goal_lm
    cells = []
    for value in range(3):
        m = directed & (relation == value)
        cells += [f"{silent[m].float().mean():.2f}", f"{hit[m].float().mean():.2f}"]
    print(f"| {run} | {colour} ({refs.count(colour)} of {len(symbols['table'])} words) | " + " | ".join(cells) + " |")
