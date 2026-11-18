"""Is the landmark carried by WHEN words are said? Decode its colour from the steps that carry a word.

    python analysis/timing_code.py runs/ablations/abl_vocab_weight/w02_s2 [more run directories ...]

10,000 held-out worlds; a lookup fitted on even worlds and scored on odd ones, without and with the
12-class landmark layout. Compare with the word-set decoder in eval.json. Prints a Markdown table.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emlang.evaluate import goal_factors, landmark_layout, play, test_worlds
from emlang.train import load

print("| Run | Timing patterns in use | Colour from timing | from timing plus layout | Colour from the set of words | "
      "from words plus layout |")
print("|---|---:|---:|---:|---:|---:|")
for run in sys.argv[1:]:
    policy, tcfg = load(run)
    state = test_worlds(tcfg.env, 10_000)
    symbols = play(policy, tcfg.env, state).symbols                                   # B,T,N
    pattern = ((symbols != 0).long() * (2 ** torch.arange(symbols.shape[1])).view(1, -1, 1)).sum(1)
    colour = goal_factors(state)["landmark"]
    layout = landmark_layout(state).unsqueeze(1).expand_as(colour)
    even = (torch.arange(len(colour)) % 2 == 0).unsqueeze(1).expand_as(colour)

    def accuracy(code):
        uniq, inverse = torch.unique(code, return_inverse=True)
        counts = torch.zeros(len(uniq), 3).index_put_((inverse[even], colour[even]), torch.ones(int(even.sum())),
                                                      accumulate=True)
        return (counts.argmax(1)[inverse[~even]] == colour[~even]).float().mean().item()

    words = json.loads((Path(run) / "eval.json").read_text())["symbols"]["decode"]["landmark"]
    print(f"| {run} | {len(torch.unique(pattern))} | {accuracy(pattern):.3f} | {accuracy(pattern * 12 + layout):.3f} | "
          f"{words['accuracy']:.3f} | {words['accuracy_given_landmark_layout']:.3f} |")
