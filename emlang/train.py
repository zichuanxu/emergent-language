"""End-to-end training by backpropagation through the differentiable world.

Every iteration samples a fresh batch of worlds, unrolls all agents with relaxed
(Gumbel-Softmax) utterances and differentiates the total return:

    physical reward + w_goal * auxiliary goal prediction
                    + w_vocab * Dirichlet-process vocabulary reward - w_utter * utterance cost

Several independent replicas (different seeds and loss weights, same world and network sizes)
train in one process: their parameters are stacked and the policy is vectorised over them, each
replica gets its own 1024 worlds, its own loss, gradient clipping and Adam statistics.
"""

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.func import functional_call, stack_module_state, vmap

from .policy import SILENCE, Policy, PolicyConfig, gumbel_sample
from .world import (AUX_DIM, GOTO, LOOKAT, NOTHING, EnvConfig, State, goal_errors, observe, peer_goal_targets,
                    reward, sample, step)

A3 = (GOTO, LOOKAT, NOTHING)
_BASE = [0, 1, 2, 5, 6, 7]  # action and addressee colour
AUX_DIMS = {"speaker": _BASE + [11, 12], "listener": _BASE + [3, 4], "both": _BASE + [3, 4, 11, 12],
            "color": _BASE + [8, 9, 10], "none": _BASE}
ACTIVE_THRESHOLD = 0.005  # a symbol is active when it fills at least 0.5% of utterance slots
CASES = {
    # verbal conditions of Figure 6 / Table 1: agents x actions x landmarks
    "1x1x3": EnvConfig(),
    "1x2x3": EnvConfig(actions=(GOTO, LOOKAT)),
    "3x3x3": EnvConfig(n_agents=3, actions=A3, goal_mode="perm"),
    # non-verbal conditions of Figure 7 (no utterances). One agent holds a goal. In half of the
    # episodes it is for the holder itself, so that looking at or going to a location is an ordinary
    # skill; in the other half it is "the other agent goes to this location", and the other agent
    # has no goal of its own.
    # Pointing: gaze is visible and the holder's own goals are look-at goals.
    "pointing": EnvConfig(goal_mode="oneway", oneway_self=0.5, actions=(GOTO,), self_actions=(LOOKAT,),
                          comm=False, see_agents=True, see_gaze=True),
    # Variant without self-goals: the two-agent verbal task (each agent holds a go-to goal for the
    # other) with the channel removed and positions and gaze visible. Neither agent can lead the
    # other to its target because it has to be somewhere else itself.
    "pointing_swap": EnvConfig(comm=False, see_agents=True, see_gaze=True),
    "guiding": EnvConfig(goal_mode="oneway", oneway_self=0.5, comm=False, see_agents=True),
    "pushing": EnvConfig(goal_mode="oneway", oneway_self=0.5, comm=False, see_agents=True, blind_recipient=True,
                         collisions=True, horizon=20),
}


def contexts_for(case: str, cfg: EnvConfig, mixed: bool, two_landmarks: float = 0.0) -> list[tuple[EnvConfig, float]]:
    """Training distribution over world configurations. The first entry is the main context.

    `two_landmarks` is the share of updates run in the main world with only two of the three
    coloured landmarks present. Two points have no geometry that a listener in another frame could
    use to tell them apart, so in those worlds colour is the only way to refer to a landmark.
    `mixed` adds simpler worlds: one landmark, fewer action types, and for three agents two agents.
    """
    out = [(replace(cfg, n_landmarks=2), two_landmarks)] if two_landmarks > 0 else []
    rest = 1.0 - two_landmarks
    if not mixed:
        return [(cfg, rest)] + out
    simpler = [replace(cfg, n_landmarks=1)]
    simpler += [replace(cfg, actions=cfg.actions[:k]) for k in range(1, len(cfg.actions))]
    if cfg.n_agents == 3:
        simpler += [replace(cfg, n_agents=2, goal_mode="swap", actions=cfg.actions[:k]) for k in (1, 2, 3)]
    simpler = list(dict.fromkeys(simpler))
    return [(cfg, rest / 2)] + out + [(c, rest / 2 / len(simpler)) for c in simpler]


@dataclass(frozen=True)
class TrainConfig:
    case: str = "1x1x3"
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    comm: bool = True
    mixed: bool = False
    two_landmarks: float = 0.0
    iters: int = 5000
    batch: int = 1024
    lr: float = 1e-3
    grad_clip: float = 5.0
    seed: int = 1
    w_goal: float = 1.0
    w_vocab: float = 0.02
    w_utter: float = 0.005
    dp_alpha: float = 1.0
    penalty_start: int = 300   # iteration at which the two communication penalties switch on
    penalty_ramp: int = 500    # iterations over which they grow linearly to full weight
    # What the auxiliary task predicts about a speaker's goal besides action and addressee:
    #   speaker  - goal location in the speaker's frame (the literal goal vector g_j)
    #   listener - goal location in the listener's frame
    #   both     - both locations
    #   color    - colour of the landmark at the goal location (privileged label)
    #   none     - nothing about the location
    aux_frame: str = "listener"


# Fields that may differ between replicas trained in one process.
PER_REPLICA = ("seed", "w_goal", "w_vocab", "w_utter", "dp_alpha", "penalty_start", "penalty_ramp", "grad_clip",
               "aux_frame")


class Ensemble(torch.nn.Module):
    """R independent policies evaluated in one vectorised call.

    Inputs carry R*B worlds, replica-major: rows [r*B, (r+1)*B) belong to replica r.
    """

    def __init__(self, policies: list[Policy]):
        super().__init__()
        self.cfg = policies[0].cfg
        self.replicas = len(policies)
        self._template = [policies[0]]  # not registered: its own weights are unused
        params, _ = stack_module_state(policies)
        self.names = list(params)
        self.stacked = torch.nn.ParameterList([torch.nn.Parameter(params[name].detach().clone()) for name in self.names])

    def initial_memory(self, batch: int, n: int, device):
        return self._template[0].initial_memory(batch, n, device)

    def forward(self, ent, mask, goal, heard, memory, comm: bool = True):
        r = self.replicas
        split = lambda x: x.reshape(r, x.shape[0] // r, *x.shape[1:])
        merge = lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        template = self._template[0]
        template.train(self.training)

        def one(params, ent, mask, goal, heard, mem_c, mem_a):
            move, gaze, logits, pred, (mem_c, mem_a) = functional_call(
                template, params, (ent, mask, goal, heard, (mem_c, mem_a)), {"comm": comm})
            return move, gaze, logits, pred, mem_c, mem_a

        out = vmap(one, randomness="different")(
            dict(zip(self.names, self.stacked)), split(ent), split(mask), split(goal), split(heard),
            split(memory[0]), split(memory[1]))
        move, gaze, logits, pred, mem_c, mem_a = (merge(x) for x in out)
        return move, gaze, logits, pred, (mem_c, mem_a)

    def replica_state(self, index: int) -> dict:
        return {name: p.detach()[index].clone() for name, p in zip(self.names, self.stacked)}

    def clip_grad_norm_(self, max_norm: Tensor) -> Tensor:
        """Per-replica gradient clipping. max_norm: R (0 disables clipping for that replica)."""
        grads = [p.grad for p in self.stacked if p.grad is not None]  # unused modules have no gradient
        norm = sum(g.reshape(self.replicas, -1).square().sum(1) for g in grads).sqrt()
        scale = torch.where(max_norm > 0, (max_norm / (norm + 1e-6)).clamp(max=1.0), torch.ones_like(norm))
        for g in grads:
            g.mul_(scale.view(-1, *[1] * (g.dim() - 1)))
        return norm


@dataclass
class Episode:
    reward: Tensor            # B,T   shared physical reward after each transition
    goal_err: Tensor | None   # B,T,N squared error of every goal
    aux_err: Tensor           # B,AUX_DIM squared auxiliary prediction error per target dimension, mean over pairs
    probs: Tensor             # B,T,N,K utterance probabilities
    final: State
    symbols: Tensor | None = None   # B,T,N emitted symbol index
    pos: Tensor | None = None       # B,T+1,N,2
    gaze: Tensor | None = None      # B,T+1,N,2
    vel: Tensor | None = None

    def aux(self, frame: str) -> Tensor:
        """Auxiliary reward B for one choice of target."""
        return -self.aux_err[:, AUX_DIMS[frame]].sum(-1)


def rollout(policy, cfg: EnvConfig, state: State, *, hard: bool, comm: bool | None = None,
            record: bool = False, heard_hook=None, obs_hook=None, tf32: bool = False) -> Episode:
    """Unroll one batch. Utterances emitted at step t are heard at step t+1.

    tf32 lets CUDA use TF32 matrix products inside the policy network only; observation, physics
    and reward always run in full float32."""
    comm = cfg.comm if comm is None else comm
    b, n = state.pos.shape[:2]
    k = policy.cfg.vocab
    dev = state.pos.device
    utter = torch.zeros(b, n, k, device=dev)
    utter[..., SILENCE] = 1.0
    memory = policy.initial_memory(b, n, dev)
    rewards, errs, probs, symbols = [], [], [], []
    pos, gaze, vel = [state.pos], [state.gaze], [state.vel]
    pred = before = None
    for t in range(cfg.horizon):
        ent, mask, goal = observe(cfg, state)
        heard = utter.unsqueeze(1).expand(b, n, n, k)
        if heard_hook is not None:
            heard = heard_hook(t, state, heard)
        if obs_hook is not None:
            ent, mask, goal = obs_hook(t, state, ent, mask, goal)
        with _tf32(tf32):
            move, look, logits, pred, memory = policy(ent, mask, goal, heard, memory, comm=comm)
        if comm:
            utter = gumbel_sample(logits, policy.cfg.temperature, hard, policy.cfg.straight_through)
            probs.append(logits.softmax(-1))
        else:
            probs.append(utter)
        symbols.append(utter.argmax(-1))
        before, state = state, step(cfg, state, move, look)
        rewards.append(reward(state))
        if record:
            errs.append(goal_errors(state))
            pos.append(state.pos), gaze.append(state.gaze), vel.append(state.vel)

    if comm and n > 1:
        # At the end of the episode every listener i predicts every other speaker j's goal.
        error = (pred - peer_goal_targets(before).detach()).square()
        off_diag = ~torch.eye(n, dtype=torch.bool, device=dev)
        aux_err = (error * off_diag.unsqueeze(-1)).sum((1, 2)) / (n * (n - 1))
    else:
        aux_err = torch.zeros(b, AUX_DIM, device=dev)
    return Episode(
        reward=torch.stack(rewards, 1), goal_err=torch.stack(errs, 1) if record else None, aux_err=aux_err,
        probs=torch.stack(probs, 1), final=state, symbols=torch.stack(symbols, 1),
        pos=torch.stack(pos, 1) if record else None, gaze=torch.stack(gaze, 1) if record else None,
        vel=torch.stack(vel, 1) if record else None,
    )


class _tf32:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            torch.backends.cuda.matmul.allow_tf32 = True

    def __exit__(self, *_):
        if self.enabled:
            torch.backends.cuda.matmul.allow_tf32 = False


def vocabulary_reward(probs: Tensor, alpha: float | Tensor = 1.0) -> tuple[Tensor, Tensor]:
    """Dirichlet-process log-likelihood of the uttered (non-silence) symbols, per utterance slot.

    r_c = sum_k n_k log(n_k / (alpha + n - 1)), with expected counts n_k accumulated over
    agents, time steps and batch entries. probs is ...,K with any leading shape, or R,...,K with
    one value per replica when alpha is a tensor of R values.
    Returns (r_c / slots, fraction of non-silent slots).
    """
    if not torch.is_tensor(alpha):
        probs, alpha = probs.unsqueeze(0), torch.tensor([float(alpha)], device=probs.device)
        squeeze = True
    else:
        squeeze = False
    flat = probs.reshape(probs.shape[0], -1, probs.shape[-1])
    slots = flat.shape[1]
    counts = flat.sum(1)[:, SILENCE + 1:]
    total = counts.sum(1, keepdim=True)
    logp = (counts.clamp_min(1e-8) / (alpha.unsqueeze(1) + total - 1).clamp_min(1e-8)).log()
    out = (counts * logp).sum(1) / slots, total.squeeze(1) / slots
    return (out[0][0], out[1][0]) if squeeze else out


def penalty_factor(cfg: TrainConfig, it: int) -> float:
    return float(min(1.0, max(0.0, (it - cfg.penalty_start) / max(1, cfg.penalty_ramp))))


LOG_NAMES = ("reward", "terminal", "aux", "vocab", "utter", "factor", "grad_norm")


def train(cfgs: TrainConfig | list[TrainConfig], run_dir: Path, device: str = "cpu", log_every: int = 100,
          threads: int = 2, names: list[str] | None = None, tf32: bool = False):
    """Train one replica per configuration. Writes run_dir/<name>/{config.json,init.pt,final.pt,metrics.npz}
    (directly into run_dir for a single unnamed replica). Resumes from run_dir/ckpt.pt if present; on
    SIGTERM/SIGINT it checkpoints after the current iteration and exits with code 75."""
    single = isinstance(cfgs, TrainConfig)
    cfgs = [cfgs] if single else list(cfgs)
    shared = [replace(c, **{f: getattr(cfgs[0], f) for f in PER_REPLICA}) for c in cfgs]
    if any(c != shared[0] for c in shared):
        raise ValueError(f"replicas may differ only in {PER_REPLICA}")
    base, r = cfgs[0], len(cfgs)
    names = names or ([""] if single else [f"r{i}" for i in range(r)])
    dirs = [run_dir / name if name else run_dir for name in names]
    torch.set_num_threads(threads)
    run_dir.mkdir(parents=True, exist_ok=True)
    contexts = contexts_for(base.case, base.env, base.mixed, base.two_landmarks)
    weights = torch.tensor([w for _, w in contexts])

    singles = []
    for cfg in cfgs:  # replica r starts from exactly the weights a lone run with its seed would have
        torch.manual_seed(cfg.seed)
        singles.append(Policy(cfg.policy))
    policy = Ensemble(singles).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=base.lr)
    torch.manual_seed(base.seed + 7919 * r)
    gen = torch.Generator().manual_seed(base.seed + 1000)
    k = base.policy.vocab
    per = lambda f: torch.tensor([float(getattr(c, f)) for c in cfgs], device=device)
    w_goal, w_vocab, w_utter, alpha, clip = (per(f) for f in ("w_goal", "w_vocab", "w_utter", "dp_alpha", "grad_clip"))
    aux_mask = torch.zeros(r, AUX_DIM, device=device)
    for i, cfg in enumerate(cfgs):
        aux_mask[i, AUX_DIMS[cfg.aux_frame]] = 1.0

    ctx_log: list[int] = []
    log = {name: np.zeros((base.iters, r), dtype=np.float32) for name in LOG_NAMES}
    freq_log = np.zeros((base.iters, r, k), dtype=np.float32)
    first = 1
    ckpt_path = run_dir / "ckpt.pt"
    cfg_dicts = [asdict(c) for c in cfgs]
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        if [{**c, "iters": base.iters} for c in saved["configs"]] != cfg_dicts:
            raise ValueError(f"{ckpt_path} was written with a different configuration")
        policy.load_state_dict(saved["model"]), opt.load_state_dict(saved["opt"])
        # generator states are CPU byte tensors; map_location may have moved them to the training device
        gen.set_state(saved["gen"].cpu()), torch.set_rng_state(saved["rng"].cpu())
        if saved.get("cuda_rng") is not None and torch.device(device).type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
        done = saved["iter"]
        ctx_log = saved["ctx"]
        for name in LOG_NAMES:
            log[name][:done] = saved["log"][name]
        freq_log[:done] = saved["freq"]
        first = done + 1
        print(f"resuming at iteration {first}", flush=True)
    for i, d in enumerate(dirs):
        d.mkdir(parents=True, exist_ok=True)
        # rewritten on resume: the number of updates may have been raised
        (d / "config.json").write_text(json.dumps(
            {"train": cfg_dicts[i], "contexts": [{"env": asdict(c), "p": w} for c, w in contexts],
             "torch": torch.__version__, "device": device, "replicas_in_process": r,
             "tf32_in_policy_network": tf32}, indent=1))
        if first == 1:
            torch.save({"model": policy.replica_state(i), "config": cfg_dicts[i], "iter": 0}, d / "init.pt")

    stop = {"requested": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(requested=True))

    def write_metrics(it):
        for i, d in enumerate(dirs):
            np.savez_compressed(d / "metrics.npz", ctx=np.asarray(ctx_log), freq=freq_log[:it, i],
                                **{name: log[name][:it, i] for name in LOG_NAMES})

    def checkpoint(it):
        tmp = run_dir / "ckpt.pt.tmp"
        torch.save({"model": policy.state_dict(), "opt": opt.state_dict(), "configs": cfg_dicts, "iter": it,
                    "names": names,
                    "gen": gen.get_state(), "rng": torch.get_rng_state(), "ctx": ctx_log,
                    "cuda_rng": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
                    "log": {name: log[name][:it] for name in LOG_NAMES}, "freq": freq_log[:it]}, tmp)
        os.replace(tmp, ckpt_path)
        write_metrics(it)

    start = time.time()
    for it in range(first, base.iters + 1):
        ctx = int(torch.multinomial(weights, 1, generator=gen))
        env = contexts[ctx][0]
        use_comm = base.comm and env.comm
        state = sample(env, base.batch * r, gen, device)
        policy.train()
        ep = rollout(policy, env, state, hard=False, comm=use_comm, tf32=tf32)
        rep = lambda x: x.reshape(r, base.batch, *x.shape[1:])
        phys = rep(ep.reward).mean((1, 2))
        aux = -(rep(ep.aux_err).mean(1) * aux_mask).sum(1)
        factor = torch.tensor([penalty_factor(c, it) for c in cfgs], device=device)
        if use_comm:
            vocab, utter = vocabulary_reward(rep(ep.probs), alpha)
        else:
            vocab = utter = torch.zeros(r, device=device)
        loss = -phys - w_goal * aux - factor * (w_vocab * vocab - w_utter * utter)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"non-finite loss at iteration {it}: {loss.tolist()}")
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        grad_norm = policy.clip_grad_norm_(clip)
        opt.step()

        with torch.no_grad():
            hard = torch.distributions.Categorical(probs=ep.probs).sample() if use_comm else ep.symbols
            onehot = torch.nn.functional.one_hot(rep(hard).reshape(r, -1), k).float().mean(1)
            freq_log[it - 1] = onehot.cpu().numpy()
            values = (phys, rep(ep.reward)[:, :, -1].mean(1), aux, vocab, utter, factor, grad_norm)
            for name, value in zip(LOG_NAMES, values):
                log[name][it - 1] = value.detach().cpu().numpy()
        ctx_log.append(ctx)
        if it % log_every == 0 or it == base.iters:
            main = [i for i in range(max(0, it - 200), it) if ctx_log[i] == 0]
            for i, name in enumerate(names):
                mean = lambda key: round(float(log[key][main, i].mean()), 4) if main else None
                active = int((freq_log[main[-20:], i].mean(0)[1:] >= ACTIVE_THRESHOLD).sum()) if main else -1
                print(json.dumps({"run": name, "it": it, "reward": mean("reward"), "terminal": mean("terminal"),
                                  "aux": mean("aux"), "utter": mean("utter"), "active": active,
                                  "sec": round(time.time() - start)}), flush=True)
        if it % 250 == 0 or it == base.iters or stop["requested"]:
            checkpoint(it)
        if stop["requested"] and it < base.iters:
            print(f"stop requested: checkpointed at iteration {it}, exit 75", flush=True)
            sys.exit(75)
    for i, d in enumerate(dirs):
        torch.save({"model": policy.replica_state(i), "config": cfg_dicts[i], "iter": base.iters}, d / "final.pt")
    write_metrics(base.iters)
    return policy


def config_from_dict(d: dict) -> TrainConfig:
    env = dict(d["env"])
    env["actions"] = tuple(env["actions"])
    env["self_actions"] = tuple(env.get("self_actions", ()))
    return TrainConfig(**{**d, "env": EnvConfig(**env), "policy": PolicyConfig(**d["policy"])})


def load(run_dir: Path | str, which: str = "final", device: str = "cpu") -> tuple[Policy, TrainConfig]:
    """Load one replica's final.pt or init.pt as an ordinary single Policy."""
    saved = torch.load(Path(run_dir) / f"{which}.pt", map_location=device, weights_only=True)
    cfg = config_from_dict(saved["config"])
    policy = Policy(cfg.policy).to(device)
    policy.load_state_dict(saved["model"])
    return policy.eval(), cfg


def export_checkpoint(run_dir: Path | str):
    """Write final.pt for every replica from an unfinished run's ckpt.pt (to inspect a run in progress)."""
    run_dir = Path(run_dir)
    saved = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    template = Policy(config_from_dict(saved["configs"][0]).policy)
    keys = list(dict(template.named_parameters()))
    for i, name in enumerate(saved["names"]):
        d = run_dir / name if name else run_dir
        state = {key: saved["model"][f"stacked.{j}"][i].clone() for j, key in enumerate(keys)}
        torch.save({"model": state, "config": saved["configs"][i], "iter": saved["iter"]}, d / "final.pt")
    return saved["iter"]


def _list(text: str, cast):
    return [cast(x) for x in str(text).split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=sorted(CASES), default="1x1x3")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--no-comm", action="store_true", help="train the no-communication baseline")
    p.add_argument("--mixed", action="store_true", help="also train on simpler world configurations")
    p.add_argument("--two-landmarks", type=float, default=0.0, help="share of updates with two landmarks")
    p.add_argument("--horizon", type=int)
    p.add_argument("--close-start", type=float, help="pushing: share of training episodes that start near contact")
    p.add_argument("--contact-margin", type=float)
    p.add_argument("--close-max", type=float, help="pushing: largest near start, in body diameters")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--tf32", action="store_true", help="CUDA TF32 matrix products inside the policy network")
    p.add_argument("--names", help="comma-separated replica directory names (default r0,r1,...)")
    p.add_argument("--export-checkpoint", action="store_true", help="only write final.pt files from ckpt.pt")
    d = TrainConfig()
    for name in ("iters", "batch"):
        p.add_argument("--" + name, type=int, default=getattr(d, name))
    p.add_argument("--lr", type=float, default=d.lr)
    replica_help = "one value, or a comma-separated list with one value per replica"
    for name in PER_REPLICA:
        p.add_argument("--" + name.replace("_", "-"), default=str(getattr(d, name)), help=replica_help)
    pd = PolicyConfig()
    for name in ("hidden", "feature", "memory", "vocab"):
        p.add_argument("--" + name, type=int, default=getattr(pd, name))
    for name in ("dropout", "temperature", "move_scale", "action_noise", "memory_noise", "logit_scale", "goal_gain"):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(pd, name))
    p.add_argument("--init", choices=("default", "he", "xavier"), default=pd.init)
    p.add_argument("--straight-through", action="store_true",
                   help="train on one-hot utterances with the relaxed sample's gradient")
    p.add_argument("--aux-ungrounded", action="store_true",
                   help="auxiliary head reads the utterance stream only (literal Figure 3)")
    a = p.parse_args()
    if a.export_checkpoint:
        print("exported iteration", export_checkpoint(a.run_dir))
        return
    env = CASES[a.case]
    for name in ("horizon", "close_start", "contact_margin", "close_max"):
        if getattr(a, name) is not None:
            env = replace(env, **{name: getattr(a, name)})
    casts = {"seed": int, "penalty_start": int, "penalty_ramp": int, "aux_frame": str}
    lists = {name: _list(getattr(a, name), casts.get(name, float)) for name in PER_REPLICA}
    r = max(len(v) for v in lists.values())
    if any(len(v) not in (1, r) for v in lists.values()):
        p.error("per-replica lists must all have the same length")
    lists = {name: v * r if len(v) == 1 else v for name, v in lists.items()}
    if any(frame not in AUX_DIMS for frame in lists["aux_frame"]):
        p.error(f"aux-frame must be one of {sorted(AUX_DIMS)}")
    if not 0.0 <= a.two_landmarks < 1.0:
        p.error("--two-landmarks must be a share below 1: the main world needs some updates")
    policy_cfg = PolicyConfig(
        hidden=a.hidden, feature=a.feature, memory=a.memory, vocab=a.vocab, dropout=a.dropout,
        temperature=a.temperature, move_scale=a.move_scale, action_noise=a.action_noise, memory_noise=a.memory_noise,
        aux_grounded=not a.aux_ungrounded, logit_scale=a.logit_scale, goal_gain=a.goal_gain, init=a.init,
        straight_through=a.straight_through)
    cfgs = [TrainConfig(case=a.case, env=env, comm=not a.no_comm, mixed=a.mixed, two_landmarks=a.two_landmarks, policy=policy_cfg, iters=a.iters,
                        batch=a.batch, lr=a.lr, **{name: lists[name][i] for name in PER_REPLICA}) for i in range(r)]
    names = a.names.split(",") if a.names else None
    if names and len(names) != r:
        p.error("--names needs one name per replica")
    if r == 1 and not names:
        train(cfgs[0], a.run_dir, a.device, threads=a.threads, tf32=a.tf32)
    else:
        train(cfgs, a.run_dir, a.device, threads=a.threads, names=names, tf32=a.tf32)


if __name__ == "__main__":
    main()
