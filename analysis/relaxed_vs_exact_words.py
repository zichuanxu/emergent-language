"""Active words of one policy when listeners hear relaxed messages (as in training) or exact symbols (as at test).

    python analysis/relaxed_vs_exact_words.py runs/main/1x2x3_comm/s4 [more run directories ...]

Same 4,000 held-out worlds and the same random seed in every condition. Prints a Markdown table.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emlang.evaluate import test_worlds
from emlang.train import load, rollout

print("| Run | Dropout | Messages heard | Words above 0.5% of slots | above 1% | Non-silent share | Terminal reward |")
print("|---|---|---|---:|---:|---:|---:|")
for run in sys.argv[1:]:
    policy, tcfg = load(run)
    state = test_worlds(tcfg.env, 4000)
    for mode in ("eval", "train"):
        for hard in (True, False):
            getattr(policy, mode)()
            torch.manual_seed(0)
            with torch.no_grad():
                ep = rollout(policy, tcfg.env, state, hard=hard, record=True)
                symbols = torch.distributions.Categorical(probs=ep.probs).sample()
            share = torch.bincount(symbols.flatten(), minlength=policy.cfg.vocab).float() / symbols.numel()
            print(f"| {run} | {'on' if mode == 'train' else 'off'} | {'exact' if hard else 'relaxed'} | "
                  f"{int((share[1:] >= 0.005).sum())} | {int((share[1:] >= 0.01).sum())} | {1 - share[0].item():.3f} | "
                  f"{ep.reward[:, -1].mean().item():.3f} |")
