"""Learnability probes that isolate the two halves of landmark reference.

    python -m emlang.probe speaker  [--goal-gain 1] ...   # supervised: name the colour at the goal location
    python -m emlang.probe listener ...                   # oracle colour words in, physical reward out

Neither probe is part of the reproduction. They measure how quickly the Figure 3 architecture
can learn each half when the other half is given, which bounds how fast language can emerge.
"""

import argparse
import json
import time
from dataclasses import replace

import torch
from torch.nn import functional as F

from .policy import Policy, PolicyConfig
from .train import CASES, rollout
from .world import _take, observe, sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("speaker", "listener"))
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--goal-gain", type=float, default=1.0)
    p.add_argument("--init", default="default")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    cfg = replace(CASES["1x1x3"], horizon=a.horizon)
    policy = Policy(PolicyConfig(goal_gain=a.goal_gain, init=a.init))
    opt = torch.optim.Adam(policy.parameters(), lr=a.lr)
    gen = torch.Generator().manual_seed(a.seed + 1)
    b, k = a.batch, policy.cfg.vocab
    start = time.time()
    for it in range(1, a.iters + 1):
        s = sample(cfg, b, gen)
        color = _take(s.lm_color, s.goal_lm).argmax(-1)
        if a.mode == "listener":
            oracle = torch.zeros(b, 2, k).scatter_(-1, (color + 1).unsqueeze(-1), 1.0)
            ep = rollout(policy, cfg, s, hard=False, heard_hook=lambda t, st, heard: oracle.unsqueeze(1).expand(b, 2, 2, k))
            loss = -ep.reward.mean()
            stat = {"reward": ep.reward.mean().item(), "terminal": ep.reward[:, -1].mean().item()}
        else:
            heard = torch.zeros(b, 2, 2, k)
            heard[..., 0] = 1
            logits = policy(*observe(cfg, s), heard, policy.initial_memory(b, 2, "cpu"))[2][..., 1:4]
            loss = F.cross_entropy(logits.reshape(-1, 3), color.reshape(-1))
            stat = {"loss": loss.item(), "accuracy": (logits.argmax(-1) == color).float().mean().item()}
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        opt.step()
        if it % 100 == 0:
            print(json.dumps({"it": it, **{key: round(v, 4) for key, v in stat.items()}, "sec": round(time.time() - start)}), flush=True)


if __name__ == "__main__":
    main()
