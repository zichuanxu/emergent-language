"""Direct test on the first, failed implementation ("v1" in docs/REVIEW.md): can its policy learn,
under full supervision, to name the colour of the landmark at its private goal location?

    EMLANG_V1_ROOT=/path/to/v1 python analysis/prior_code_probe.py default|he [iters]

v1 is not part of this repository; EMLANG_V1_ROOT is the directory that holds its `src/` package.
The two logs next to this file are the outputs quoted in docs/REVIEW.md.

The emergent task needs this mapping at the speaker. `default` uses the prior code unchanged,
`he` only re-initialises its hidden layers (He-normal). Prints supervised accuracy over training.
"""
import os
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, os.environ["EMLANG_V1_ROOT"])
from src.policy import Policy, PolicyConfig  # noqa: E402
from src.world import World, make_config  # noqa: E402

mode, iters = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 600
torch.manual_seed(0)
torch.set_num_threads(4)
world = World(make_config("1x1x3"))
policy = Policy(PolicyConfig())
if mode == "he":
    for module in (policy.entity_encoder, policy.message_encoder, policy.action_encoder):
        for layer in list(module)[:-1]:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                torch.nn.init.zeros_(layer.bias)
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
gen = torch.Generator().manual_seed(1)
start = time.time()
for it in range(1, iters + 1):
    state = world.sample(1024, generator=gen)
    entities, mask, goal = world.observe(state)
    silence = torch.zeros(1024, 2, 20)
    silence[..., 0] = 1
    output, _ = policy(entities, mask, goal, silence, policy.initial_state(state.position))
    colour = state.landmark_colors.gather(1, state.goal_landmark.unsqueeze(-1).expand(-1, -1, 3)).argmax(-1)
    logits = output.logits[..., 1:4]
    loss = F.cross_entropy(logits.reshape(-1, 3), colour.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 100 == 0:
        acc = (logits.argmax(-1) == colour).float().mean().item()
        print(f"{mode} it={it} loss={loss.item():.4f} accuracy={acc:.3f} sec={time.time() - start:.0f}", flush=True)
