"""Held-out evaluation: Table 1 rewards, symbol semantics, generalization and non-verbal probes.

    python -m emlang.evaluate RUN_DIR [--baseline NOCOMM_RUN_DIR] [--trials 10000]

writes RUN_DIR/eval.json and RUN_DIR/eval_episodes.npz. All test rollouts use exact
categorical samples (no relaxation), dropout off, and worlds drawn from a seed that
training never used. Paired comparisons replay the same worlds. Runs of one policy (with messages,
with messages silenced, with an observation hidden) also replay the same random draws; a
communicating policy and its no-communication baseline do not, because only one of them samples
utterances.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .policy import SILENCE
from .train import ACTIVE_THRESHOLD, Episode, load, rollout
from .world import (ACTION_NAMES, COLOR_NAMES, GOTO, LOOKAT, NOGOAL, NOTHING, NOVEL_COLOR, EnvConfig, State,
                    _take, sample)

TEST_SEED = 20_260_000


def play(policy, cfg: EnvConfig, state: State, *, seed: int = 0, hard: bool = True, **kwargs) -> Episode:
    """One deterministic-in-seed evaluation rollout."""
    policy.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        return rollout(policy, cfg, state, hard=hard, record=True, **kwargs)


def test_worlds(cfg: EnvConfig, trials: int, seed: int = TEST_SEED) -> State:
    return sample(cfg, trials, torch.Generator().manual_seed(seed))


def ci(x: Tensor | np.ndarray) -> list[float]:
    """Mean and 95% normal interval over episodes (conditional on the checkpoint)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return [float(x.mean()), float(x.mean() - half), float(x.mean() + half), int(len(x))]


def silence_hook(_t, _state, heard):
    out = torch.zeros_like(heard)
    out[..., SILENCE] = 1.0
    return out


# ---------------------------------------------------------------- Table 1

def reward_summary(policy, cfg: EnvConfig, state: State, **kwargs) -> dict:
    """Physical reward of one policy on fixed worlds under several reductions.

    `step_mean` is the training objective: negative squared goal error averaged over goals
    and time steps. The paper does not define its reduction, so the alternatives are listed.
    """
    ep = play(policy, cfg, state, **kwargs)
    n_goals = (state.goal_type != NOGOAL).sum(-1)
    return {
        "step_mean": ci(ep.reward.mean(1)),
        "terminal": ci(ep.reward[:, -1]),
        "step_mean_sum_over_goals": ci(ep.reward.mean(1) * n_goals),
        "terminal_sum_over_goals": ci(ep.reward[:, -1] * n_goals),
        "terminal_goal_distance": ci(final_goal_distance(ep, state)),
    }


def final_goal_distance(ep: Episode, state: State) -> Tensor:
    """Mean Euclidean miss of spatial goals at the end of the episode, per world."""
    spatial = state.goal_type <= LOOKAT
    dist = ep.goal_err[:, -1].clamp_min(0).sqrt() * spatial
    return dist.sum(-1) / spatial.sum(-1).clamp_min(1)


def landmark_choice(ep: Episode, state: State) -> dict:
    """For go-to goals addressed to another agent: which landmark does the addressee end nearest to?"""
    final = _take(ep.pos[:, -1], state.goal_rcpt)                                  # B,N,2 addressee of goal i
    nearest = (final.unsqueeze(2) - state.lm_pos.unsqueeze(1)).norm(dim=-1).argmin(-1)
    n = state.pos.shape[1]
    directed = (state.goal_type == GOTO) & (state.goal_rcpt != torch.arange(n))
    per_world = ((nearest == state.goal_lm) & directed).float().sum(1) / directed.float().sum(1).clamp_min(1)
    return {"addressee_ends_nearest_to_target_landmark": ci(per_world[directed.any(1)]),
            "chance": 1.0 / state.lm_pos.shape[1]}


def centroid_stats(ep: Episode, state: State) -> dict:
    centroid = state.lm_pos.mean(1, keepdim=True)
    return {"initial_centroid_distance": ci((ep.pos[:, 0] - centroid).norm(dim=-1).mean(-1)),
            "final_centroid_distance": ci((ep.pos[:, -1] - centroid).norm(dim=-1).mean(-1))}


# ---------------------------------------------------------------- symbol semantics

def goal_factors(state: State) -> dict[str, Tensor]:
    """Categorical description of every speaker's private goal, each B,N."""
    n = state.pos.shape[1]
    lm_color = _take(state.lm_color, state.goal_lm).argmax(-1)
    # rank of the target among the landmarks by distance from their centroid: a colour-free,
    # rotation-invariant way to single out a landmark that a private frame does not rule out
    from_centroid = (state.lm_pos - state.lm_pos.mean(1, keepdim=True)).norm(dim=-1)
    rank = from_centroid.argsort(-1).argsort(-1).gather(1, state.goal_lm)
    out = {
        "centroid_rank": torch.where(state.goal_type <= LOOKAT, rank, -1),
        "action": state.goal_type,
        "landmark": torch.where(state.goal_type <= LOOKAT, lm_color, -1),
        "addressee": _take(state.agent_color, state.goal_rcpt).argmax(-1),
        "self_goal": (state.goal_rcpt == torch.arange(n)).long(),
    }
    # Reference relative to one coloured landmark: the target is that landmark (0), or the nearer (1)
    # or farther (2) of the others from it. Both agents can evaluate this in their own frames.
    for c, color in enumerate(COLOR_NAMES):
        is_ref = state.lm_color.argmax(-1) == c                                   # B,M
        if state.lm_pos.shape[1] != 3 or not bool(is_ref.any(1).all()):
            continue
        ref = (state.lm_pos * is_ref.unsqueeze(-1)).sum(1, keepdim=True)
        from_ref = (state.lm_pos - ref).norm(dim=-1)
        relation = from_ref.argsort(-1).argsort(-1).gather(1, state.goal_lm)
        out[f"relative_to_{color}"] = torch.where(state.goal_type <= LOOKAT, relation, -1)
    return out


def landmark_layout(state: State) -> Tensor:
    """One of 12 classes per world that every agent can compute in its own frame from three
    distinctly coloured landmarks: handedness of the red-green-blue triangle and the order of its
    side lengths. Private frames are rotations, so both are shared knowledge."""
    order = state.lm_color.argmax(-1).argsort(-1)
    p = state.lm_pos.gather(1, order.unsqueeze(-1).expand(-1, -1, 2))               # B,3,2 as R,G,B
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    handed = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] > 0).long()
    sides = torch.stack(((p[:, 0] - p[:, 1]).norm(dim=-1), (p[:, 1] - p[:, 2]).norm(dim=-1),
                         (p[:, 2] - p[:, 0]).norm(dim=-1)), 1)
    rank = sides.argsort(-1)
    return handed * 6 + rank[:, 0] * 2 + (rank[:, 1] > rank[:, 2]).long()


# Labels of the values of every goal factor. A landmark can be named by its colour or by where it
# sits: nearest/farthest from the landmarks' centroid, or the nearer/farther of the other two from
# one coloured landmark ("relative_to_RED" value 0 is the red landmark itself, already a colour word).
FACTOR_VALUES = {"action": ACTION_NAMES, "landmark": COLOR_NAMES,
                 "centroid_rank": ("MOST-CENTRAL", "MID-CENTRAL", "LEAST-CENTRAL"),
                 "addressee": tuple(c + "-AGENT" for c in COLOR_NAMES),
                 **{f"relative_to_{c}": (c, f"NEARER-TO-{c}", f"FARTHER-FROM-{c}") for c in COLOR_NAMES}}
POSITION_LABELS = tuple(FACTOR_VALUES["centroid_rank"]) + tuple(
    label for c in COLOR_NAMES for label in FACTOR_VALUES[f"relative_to_{c}"][1:])
WORD_CLASSES = {"action": ACTION_NAMES, "landmark": COLOR_NAMES, "addressee": FACTOR_VALUES["addressee"],
                "position": POSITION_LABELS}
OWN_GOAL = "OWN-GOAL"                    # label of a word said (almost) only about the speaker's own goal
CLEAR_PRECISION, CLEAR_LIFT = 0.8, 0.2   # a word "means" a value if it marks it this purely, this far above its prior
MIN_CELL = 5                             # fitted streams a word set needs before its own majority is trusted


def is_clear(row: dict) -> bool:
    return row["precision"] >= CLEAR_PRECISION and row["lift"] >= CLEAR_LIFT


def analyze_symbols(symbols: Tensor, state: State, vocab: int, n_agent_colors_matter: bool) -> dict:
    """Relate emitted symbols to the speaker's goal.

    symbols: B,T,N. A stream is one speaker in one episode. For every active symbol we report
    how purely its presence in a stream indicates one value of one goal factor (precision),
    how often that value is expressed with it (recall), and when it is first uttered. Candidate
    meanings are the action, the landmark's colour, the landmark's position among the landmarks and
    (with three agents) the addressee's colour. Only streams about a goal for another agent count,
    except that a word said almost only in streams about the speaker's own goal is labelled as such.
    Decodability fits stream-content -> factor on even episodes and scores on odd episodes.
    """
    b, t, n = symbols.shape
    streams = symbols.permute(0, 2, 1).reshape(b * n, t)
    present = torch.zeros(b * n, vocab, dtype=torch.bool).scatter_(1, streams, True)
    present[:, SILENCE] = False
    freq = torch.bincount(streams.flatten(), minlength=vocab).float() / streams.numel()
    active = [k for k in range(vocab) if k != SILENCE and freq[k] >= ACTIVE_THRESHOLD]
    factors = {name: values.reshape(-1) for name, values in goal_factors(state).items()}
    directed = factors["self_goal"] == 0
    even = (torch.arange(b * n) // n) % 2 == 0
    first_time = torch.where(streams.unsqueeze(-1) == torch.arange(vocab), torch.arange(t).view(1, t, 1), t).amin(1)

    names = ["action", "landmark"] + (["addressee"] if n_agent_colors_matter else [])
    three = state.lm_pos.shape[1] == 3
    spatial = (["centroid_rank"] + [f for f in factors if f.startswith("relative_to_")]) if three else []
    table = {}
    for k in active:
        has = present[:, k] & directed
        best = None
        for name in names + spatial:
            valid = (factors[name] >= 0) & directed
            for value, label in enumerate(FACTOR_VALUES[name]):
                if name.startswith("relative_to_") and value == 0:
                    continue                                   # the coloured landmark itself: a colour word
                is_value = (factors[name] == value) & valid
                if is_value.sum() == 0:
                    continue
                precision = (has & is_value).sum().item() / max(1, has.sum().item())
                prior = is_value.sum().item() / max(1, directed.sum().item())
                recall = (has & is_value).sum().item() / max(1, is_value.sum().item())
                lift = precision - prior
                if best is None or lift > best["lift"]:
                    best = {"label": label, "factor": name, "precision": precision, "recall_directed": recall,
                            "prior": prior, "lift": lift}
        # A word may instead mark that the goal is the speaker's own (possible with three agents).
        own_share = (present[:, k] & ~directed).sum().item() / max(1, present[:, k].sum().item())
        own_prior = (~directed).float().mean().item()
        if own_share >= CLEAR_PRECISION and own_share - own_prior > best["lift"]:
            best = {"label": OWN_GOAL, "factor": "self_goal", "precision": own_share, "recall_directed": 0.0,
                    "prior": own_prior, "lift": own_share - own_prior}
        best["share_of_its_streams_about_own_goal"] = own_share
        best["slot_share"] = freq[k].item()
        best["stream_share"] = present[:, k].float().mean().item()
        best["mean_first_step"] = first_time[present[:, k], k].float().mean().item()
        table[k] = best

    # Assign each symbol the concept it marks most purely; flag unclear ones.
    labels = {k: (row["label"] if is_clear(row) else f"S{k}") for k, row in table.items()}

    def lookup(code, name, fit, score):
        """Fit code -> most frequent factor value on `fit`, score on `score` (for codes with few values)."""
        n_values = len(FACTOR_VALUES[name])
        uniq, inverse = torch.unique(code, return_inverse=True)
        counts = torch.zeros(len(uniq), n_values).index_put_(
            (inverse[fit], factors[name][fit]), torch.ones(int(fit.sum())), accumulate=True)
        guess = counts.argmax(1)[inverse[score]]
        return (guess == factors[name][score]).float().mean().item()

    def from_words(name, fit, score, context=None):
        """Decode a factor from the set of words in a stream, optionally within a context class.

        A word set seen at least MIN_CELL times in the fitted half (in that context) is decoded by
        its majority value, which also captures meanings carried by combinations of words. Rarer
        sets fall back to a naive-Bayes vote of the individual words, so that words without
        meaning cannot dilute the others. Returns (accuracy, share of scored streams on the fallback).
        """
        n_values, y = len(FACTOR_VALUES[name]), factors[name]
        ctx = torch.zeros(b * n, dtype=torch.long) if context is None else context
        n_ctx = int(ctx.max()) + 1
        uniq, inverse = torch.unique(code * n_ctx + ctx, return_inverse=True)
        ones = torch.ones(int(fit.sum()))
        counts = torch.zeros(len(uniq), n_values).index_put_((inverse[fit], y[fit]), ones, accumulate=True)
        exact, seen = counts.argmax(1)[inverse], counts.sum(1)[inverse] >= MIN_CELL
        x = present[:, 1:].float()
        group = ctx * n_values + y.clamp_min(0)
        total = torch.zeros(n_ctx * n_values).index_add_(0, group[fit], ones)
        p = (torch.zeros(n_ctx * n_values, vocab - 1).index_add_(0, group[fit], x[fit]) + 1) / (total.unsqueeze(1) + 2)
        scores = x @ (p.log() - (1 - p).log()).T + (1 - p).log().sum(1) + (total + 1).log()       # streams, groups
        bayes = scores.view(b * n, n_ctx, n_values)[torch.arange(b * n), ctx].argmax(1)
        guess = torch.where(seen, exact, bayes)
        return (guess == y)[score].float().mean().item(), (~seen)[score].float().mean().item()

    decode = {}
    code = (present[:, 1:].long() * (2 ** torch.arange(vocab - 1))).sum(1)
    layout = landmark_layout(state).repeat_interleave(n) if three else None
    for name in names + spatial:
        valid = (factors[name] >= 0) & directed
        if name == "landmark" or name in spatial:
            valid = valid & (factors["action"] <= LOOKAT)
        fit, score = valid & even, valid & ~even
        if fit.sum() == 0 or score.sum() == 0:
            continue
        majority = torch.bincount(factors[name][fit], minlength=len(FACTOR_VALUES[name])).argmax()
        accuracy, fallback = from_words(name, fit, score)
        decode[name] = {"accuracy": accuracy, "share_decoded_by_single_word_fallback": fallback,
                        "majority_baseline": (factors[name][score] == majority).float().mean().item(),
                        "streams": int(score.sum())}
        if name == "landmark" and three:
            # A listener also sees the layout. Words that single out a landmark by its position
            # relative to the others (not by colour) become decodable once the layout is known.
            accuracy, fallback = from_words(name, fit, score, layout)
            decode[name]["accuracy_given_landmark_layout"] = accuracy
            decode[name]["share_decoded_by_single_word_fallback_given_layout"] = fallback
            decode[name]["accuracy_from_layout_alone"] = lookup(layout, name, fit, score)
        if name in spatial:
            # what a pure colour vocabulary would score on this factor
            decode[name]["accuracy_from_landmark_colour"] = lookup(factors["landmark"].clamp_min(0), name, fit, score)

    by_factor = {name: [k for k, row in table.items() if row["factor"] == name and is_clear(row)] for name in names}

    def first_of(group):
        if not group:
            return torch.full((b * n,), t)
        return first_time[:, group].amin(1)

    order = {}
    for a, c in (("action", "landmark"), ("action", "addressee"), ("landmark", "addressee")):
        if a in by_factor and c in by_factor and by_factor[a] and by_factor[c]:
            ta, tc = first_of(by_factor[a]), first_of(by_factor[c])
            both = (ta < t) & (tc < t) & directed
            order[f"{a}_before_{c}"] = {"share": (ta < tc)[both].float().mean().item() if both.any() else None,
                                        "streams_with_both": int(both.sum())}
    goto_first = None
    goto_symbols = [k for k, row in table.items() if row["label"] == "GOTO" and is_clear(row)]
    lm_symbols = by_factor.get("landmark", [])
    if goto_symbols and lm_symbols:
        tg, tl = first_of(goto_symbols), first_of(lm_symbols)
        both = (tg < t) & (tl < t) & directed
        goto_first = {"share": (tg < tl)[both].float().mean().item() if both.any() else None,
                      "streams_with_both": int(both.sum())}

    words = present.sum(1).float()
    spoken = (streams != SILENCE).float()
    return {
        "active_symbols": active,
        "n_active": len(active),
        "table": {str(k): row for k, row in table.items()},
        "labels": {str(k): label for k, label in labels.items()},
        "decode": decode,
        "order": order,
        "goto_before_landmark": goto_first,
        "non_silent_share_by_step": spoken.mean(0).tolist(),
        "utterances_per_stream": {"directed_goal": ci(spoken[directed].sum(1)),
                                  "self_goal": ci(spoken[~directed].sum(1)) if (~directed).any() else None},
        "distinct_words_per_stream": {"directed_goal": ci(words[directed]),
                                      "self_goal": ci(words[~directed]) if (~directed).any() else None},
        "symbol_slot_share": freq.tolist(),
    }


def verbal_eval(policy, tcfg, trials: int, baseline=None, init_policy=None) -> tuple[dict, dict]:
    cfg = tcfg.env
    state = test_worlds(cfg, trials)
    out = {"case": tcfg.case, "trials": trials}
    out["reward"] = {"communication": reward_summary(policy, cfg, state)}
    out["reward"]["communication_relaxed_symbols"] = reward_summary(policy, cfg, state, hard=False)
    out["reward"]["messages_silenced"] = reward_summary(policy, cfg, state, heard_hook=silence_hook)
    ep = play(policy, cfg, state)
    out["centroid"] = centroid_stats(ep, state)
    out["landmark_choice"] = {"communication": landmark_choice(ep, state)}
    if baseline is not None:
        out["reward"]["no_communication"] = reward_summary(baseline, cfg, state, comm=False)
        base = play(baseline, cfg, state, comm=False)
        out["reward"]["paired_gain_step_mean"] = ci(ep.reward.mean(1) - base.reward.mean(1))
        out["reward"]["paired_gain_terminal"] = ci(ep.reward[:, -1] - base.reward[:, -1])
        out["centroid_no_communication"] = centroid_stats(base, state)
        out["landmark_choice"]["no_communication"] = landmark_choice(base, state)
    silent = play(policy, cfg, state, heard_hook=silence_hook)
    out["reward"]["paired_loss_when_silenced_step_mean"] = ci(silent.reward.mean(1) - ep.reward.mean(1))
    out["landmark_choice"]["messages_silenced"] = landmark_choice(silent, state)
    out["symbols"] = analyze_symbols(ep.symbols, state, policy.cfg.vocab, cfg.n_agents > 2)
    arrays = {"symbols": ep.symbols.numpy().astype(np.int8)}
    if init_policy is not None:
        arrays["symbols_before_training"] = play(init_policy, cfg, state).symbols.numpy().astype(np.int8)
    k = 64
    arrays.update(sample_pos=ep.pos[:k].numpy(), sample_gaze=ep.gaze[:k].numpy(),
                  sample_lm_pos=state.lm_pos[:k].numpy(), sample_lm_color=state.lm_color[:k].numpy(),
                  sample_agent_color=state.agent_color[:k].numpy(), sample_goal_type=state.goal_type[:k].numpy(),
                  sample_goal_lm=state.goal_lm[:k].numpy(), sample_goal_rcpt=state.goal_rcpt[:k].numpy())
    return out, arrays


# ---------------------------------------------------------------- context sensitivity

def context_usage(policy, tcfg, trials: int, labels: dict) -> dict:
    """Do words disappear when context makes them redundant (one landmark, one action)?"""
    cfg = tcfg.env
    variants = {"full": cfg, "two_landmarks": replace(cfg, n_landmarks=2), "one_landmark": replace(cfg, n_landmarks=1)}
    if len(cfg.actions) > 1:
        variants["goto_only"] = replace(cfg, actions=(GOTO,))
    out = {}
    for name, variant in variants.items():
        state = test_worlds(variant, trials, TEST_SEED + 7)
        ep = play(policy, variant, state)
        b, t, n = ep.symbols.shape
        streams = ep.symbols.permute(0, 2, 1).reshape(b * n, t)
        directed = (goal_factors(state)["self_goal"] == 0).reshape(-1)
        present = torch.zeros(b * n, policy.cfg.vocab, dtype=torch.bool).scatter_(1, streams, True)
        share = {}
        for factor, members in WORD_CLASSES.items():
            group = [int(k) for k, lab in labels.items() if lab in members]
            if group:
                share[factor] = present[directed][:, group].any(1).float().mean().item()
        out[name] = {"share_of_directed_streams_using_word_class": share,
                     "utterances_per_directed_stream": (streams[directed] != SILENCE).float().sum(1).mean().item(),
                     "reward_step_mean": ep.reward.mean().item(), "reward_terminal": ep.reward[:, -1].mean().item()}
    return out


# ---------------------------------------------------------------- generalization probes (3-agent policy)

def _first_directed(state: State, action: int = GOTO) -> tuple[Tensor, Tensor]:
    """Per world: index of the first speaker whose goal is `action` for another agent, and validity."""
    n = state.pos.shape[1]
    ok = (state.goal_type == action) & (state.goal_rcpt != torch.arange(n))
    return ok.float().argmax(1), ok.any(1)


def _pick(x: Tensor, idx: Tensor) -> Tensor:
    return x[torch.arange(len(idx)), idx]


def _append_agent(state: State, color: Tensor, gen) -> State:
    """Add one agent with no goal. It is the last agent; its own utterances are silenced by the caller."""
    b = state.pos.shape[0]
    pos = torch.rand(b, 1, 2, generator=gen) * 2 - 1
    cat = lambda a, x: torch.cat((a, x), 1)
    return state.clone(
        pos=cat(state.pos, pos), vel=cat(state.vel, torch.zeros(b, 1, 2)), gaze=cat(state.gaze, pos),
        angle=cat(state.angle, torch.rand(b, 1, generator=gen) * 6.2832),
        agent_color=cat(state.agent_color, color.unsqueeze(1)),
        goal_type=cat(state.goal_type, torch.full((b, 1), NOGOAL)),
        goal_target=cat(state.goal_target, torch.zeros(b, 1, 2)), goal_lm=cat(state.goal_lm, torch.zeros(b, 1).long()),
        goal_rcpt=cat(state.goal_rcpt, torch.full((b, 1), state.pos.shape[1])),
        blind=cat(state.blind, torch.zeros(b, 1, dtype=torch.bool)), home=cat(state.home, pos))


def _mute_last(_t, _state, heard):
    heard = heard.clone()
    heard[:, :, -1] = 0.0
    heard[:, :, -1, SILENCE] = 1.0
    return heard


def probe_extra_agent(policy, cfg: EnvConfig, trials: int) -> dict:
    """A fourth agent that shares the addressee's color should carry out the same instruction."""
    state = test_worlds(cfg, trials, TEST_SEED + 11)
    owner, valid = _first_directed(state)
    rcpt = _pick(state.goal_rcpt, owner)
    target = _pick(state.goal_target, owner)
    gen = torch.Generator().manual_seed(TEST_SEED + 12)
    same = _append_agent(state, _pick(state.agent_color, rcpt), gen)
    novel = same.clone(agent_color=torch.cat((state.agent_color, NOVEL_COLOR.expand(len(owner), 1, 3)), 1))
    res = {}
    for name, s in (("same_color", same), ("novel_color", novel)):
        ep = play(policy, cfg, s, seed=3, heard_hook=_mute_last)
        d0 = (ep.pos[:, 0, -1] - target).norm(dim=-1)
        d1 = (ep.pos[:, -1, -1] - target).norm(dim=-1)
        res[name] = {"progress": d0 - d1, "final": d1,
                     "addressee_final": (_pick(ep.pos[:, -1], rcpt) - target).norm(dim=-1)}
    return {"episodes": int(valid.sum()),
            "clone_progress_to_target_same_color": ci(res["same_color"]["progress"][valid]),
            "clone_progress_to_target_novel_color": ci(res["novel_color"]["progress"][valid]),
            "paired_difference": ci((res["same_color"]["progress"] - res["novel_color"]["progress"])[valid]),
            "clone_final_distance_same_color": ci(res["same_color"]["final"][valid]),
            "clone_final_distance_novel_color": ci(res["novel_color"]["final"][valid]),
            "addressee_final_distance": ci(res["same_color"]["addressee_final"][valid])}


def probe_conflict(policy, cfg: EnvConfig, trials: int) -> dict:
    """Two speakers send agent 2 to two different landmarks. The paper reports that it averages them."""
    state = test_worlds(cfg, trials, TEST_SEED + 21)
    b = state.pos.shape[0]
    lm = torch.stack((torch.zeros(b).long(), torch.ones(b).long(), torch.zeros(b).long()), 1)
    both = state.clone(
        goal_type=torch.tensor([GOTO, GOTO, NOTHING]).expand(b, 3).clone(),
        goal_rcpt=torch.tensor([2, 2, 0]).expand(b, 3).clone(), goal_lm=lm, goal_target=_take(state.lm_pos, lm))
    # Controls keep one command; the other speaker instead holds a private self-goal it need not announce.
    only_a = both.clone(goal_type=torch.tensor([GOTO, NOTHING, NOTHING]).expand(b, 3).clone(),
                        goal_rcpt=torch.tensor([2, 1, 0]).expand(b, 3).clone())
    only_b = both.clone(goal_type=torch.tensor([NOTHING, GOTO, NOTHING]).expand(b, 3).clone(),
                        goal_rcpt=torch.tensor([0, 2, 1]).expand(b, 3).clone())
    a, c = state.lm_pos[:, 0], state.lm_pos[:, 1]
    mid = (a + c) / 2
    centroid = state.lm_pos.mean(1)
    out = {}
    runs = (("both", both, None), ("only_first", only_a, None), ("only_second", only_b, None),
            ("both_messages_silenced", both, silence_hook))
    for name, s, hook in runs:
        p = play(policy, cfg, s, seed=5, heard_hook=hook).pos[:, -1, 2]
        d_a, d_c, d_mid, d_cen = ((p - x).norm(dim=-1) for x in (a, c, mid, centroid))
        # position along the segment first->second: 0 = first target, 1 = second target, 0.5 = average
        axis = c - a
        along = ((p - a) * axis).sum(-1) / axis.square().sum(-1).clamp_min(1e-6)
        off = (p - a - along.unsqueeze(-1) * axis).norm(dim=-1)
        near = 0.25 * axis.norm(dim=-1)                      # "at" a point: within a quarter of the separation
        out[name] = {
            "to_first_target": ci(d_a), "to_second_target": ci(d_c), "to_midpoint": ci(d_mid),
            "to_landmark_centroid": ci(d_cen),
            "midpoint_closer_than_centroid_by": ci(d_cen - d_mid),
            "projection_along_segment": ci(along), "distance_off_segment": ci(off),
            "share_between_30_and_70_percent": float(((along > 0.3) & (along < 0.7)).float().mean()),
            "share_at_first_target": float((d_a < near).float().mean()),
            "share_at_second_target": float((d_c < near).float().mean()),
            "share_at_midpoint": float((d_mid < near).float().mean()),
            "share_at_landmark_centroid": float((d_cen < near).float().mean()),
        }
    out["separation_of_targets"] = ci((a - c).norm(dim=-1))
    return out


def _add_landmark(state: State, color: Tensor, gen) -> State:
    b = state.pos.shape[0]
    pos = torch.rand(b, 1, 2, generator=gen) * 2 - 1
    return state.clone(lm_pos=torch.cat((state.lm_pos, pos), 1),
                       lm_color=torch.cat((state.lm_color, color.unsqueeze(1)), 1))


def probe_landmarks(policy, cfg: EnvConfig, trials: int) -> dict:
    """Fourth landmark: a novel-colored distractor, or a second landmark in the target's color."""
    state = test_worlds(cfg, trials, TEST_SEED + 31)
    owner, valid = _first_directed(state)
    rcpt = _pick(state.goal_rcpt, owner)
    target = _pick(state.goal_target, owner)
    target_color = _pick(_take(state.lm_color, state.goal_lm), owner)
    gen = torch.Generator().manual_seed(TEST_SEED + 32)
    novel = _add_landmark(state, NOVEL_COLOR.expand(len(owner), 3), gen)
    twin = novel.clone(lm_color=torch.cat((state.lm_color, target_color.unsqueeze(1)), 1))
    extra = novel.lm_pos[:, -1]
    pair_mid = (target + extra) / 2
    res = {}
    for name, s in (("standard", state), ("novel_distractor", novel), ("same_color_twin", twin)):
        ep = play(policy, cfg, s, seed=7)
        p = _pick(ep.pos[:, -1], rcpt)
        nearest = (p.unsqueeze(1) - s.lm_pos).norm(dim=-1).argmin(1)
        res[name] = {"to_target": (p - target).norm(dim=-1), "to_extra": (p - extra).norm(dim=-1),
                     "to_pair_midpoint": (p - pair_mid).norm(dim=-1), "nearest_is_extra": (nearest == 3).float(),
                     "nearest_is_target": (nearest == _pick(state.goal_lm, owner)).float()}
    v = valid
    axis = extra - target
    p_twin = _pick(play(policy, cfg, twin, seed=7).pos[:, -1], rcpt)
    along = ((p_twin - target) * axis).sum(-1) / axis.square().sum(-1).clamp_min(1e-6)
    return {
        "episodes": int(v.sum()),
        "standard_final_distance_to_target": ci(res["standard"]["to_target"][v]),
        "distractor": {
            "final_distance_to_target": ci(res["novel_distractor"]["to_target"][v]),
            "final_distance_to_distractor": ci(res["novel_distractor"]["to_extra"][v]),
            "share_ending_nearest_distractor": float(res["novel_distractor"]["nearest_is_extra"][v].mean()),
            "share_ending_nearest_target": float(res["novel_distractor"]["nearest_is_target"][v].mean()),
            "paired_change_in_target_distance": ci((res["novel_distractor"]["to_target"] - res["standard"]["to_target"])[v]),
        },
        "same_color_twin": {
            "final_distance_to_target": ci(res["same_color_twin"]["to_target"][v]),
            "final_distance_to_twin": ci(res["same_color_twin"]["to_extra"][v]),
            "final_distance_to_pair_midpoint": ci(res["same_color_twin"]["to_pair_midpoint"][v]),
            "midpoint_distance_if_twin_were_novel_color": ci(res["novel_distractor"]["to_pair_midpoint"][v]),
            "projection_along_target_to_twin": ci(along[v]),
            "share_between_30_and_70_percent": float(((along > 0.3) & (along < 0.7))[v].float().mean()),
        },
    }


# ---------------------------------------------------------------- non-verbal strategies

def nonverbal_eval(policy, tcfg, trials: int) -> tuple[dict, dict]:
    """Episodes in which the goal holder (sender) must get the other agent (recipient) to a location."""
    cfg = replace(tcfg.env, close_start=0.0)       # the near-contact start is a training aid only
    state = test_worlds(cfg, trials)
    sender = (state.goal_type != NOGOAL).float().argmax(1)
    rcpt = 1 - sender
    valid = (_pick(state.goal_rcpt, sender) == rcpt) & (_pick(state.goal_type, sender) == GOTO)
    target = _pick(state.goal_target, sender)
    ep = play(policy, cfg, state, seed=1)

    def hide_gaze(_t, s, ent, mask, goal):
        ent = ent.clone()
        n = s.pos.shape[1]
        ent[:, :, :n, 4:6] = ent[:, :, :n, 4:6] * torch.eye(n).unsqueeze(-1)
        return ent, mask, goal

    def hide_sender(_t, s, ent, mask, goal):
        mask = mask.clone()
        mask[torch.arange(len(rcpt)), rcpt, sender] = False       # the recipient no longer sees the sender
        return ent * mask.unsqueeze(-1), mask, goal

    if tcfg.case.startswith("pointing"):
        ablated, what = play(policy, cfg, state, seed=1, obs_hook=hide_gaze), "sender's gaze hidden"
    elif tcfg.case == "guiding":
        ablated, what = play(policy, cfg, state, seed=1, obs_hook=hide_sender), "sender hidden from the recipient"
    else:
        ablated, what = play(policy, replace(cfg, collisions=False), state, seed=1), "contact disabled"

    def rcpt_dist(e, t):
        return (_pick(e.pos[:, t], rcpt) - target).norm(dim=-1)

    steps = ep.pos.shape[1]
    spos = torch.stack([_pick(ep.pos[:, t], sender) for t in range(steps)], 1)
    sgaze = torch.stack([_pick(ep.gaze[:, t], sender) for t in range(steps)], 1)
    to_target = target.unsqueeze(1) - spos
    # the gaze point recorded at index t was set from the position at index t - 1
    cos = torch.nn.functional.cosine_similarity(sgaze[:, 1:] - spos[:, :-1], to_target[:, :-1], dim=-1)
    sep = (ep.pos[:, :, 0] - ep.pos[:, :, 1]).norm(dim=-1)
    increase = rcpt_dist(ablated, -1) - rcpt_dist(ep, -1)
    v = valid
    own = ~valid & (_pick(state.goal_rcpt, sender) == sender)
    own_miss = _pick(ep.goal_err[:, -1], sender).clamp_min(0).sqrt()   # position or gaze miss of the holder's own goal
    out = {
        "case": tcfg.case, "trials": trials, "directed_goto_episodes": int(v.sum()), "ablation": what,
        "reward": {"step_mean": ci(ep.reward.mean(1)), "terminal": ci(ep.reward[:, -1]),
                   "terminal_directed": ci(ep.reward[:, -1][v])},
        "self_goal_final_distance": ci(own_miss[own]) if own.any() else None,
        # In self-goal episodes nobody is asked to bring the other agent anywhere. If it ends at the
        # sender's target anyway, it follows unconditionally rather than responding to being addressed.
        "bystander_initial_distance_to_senders_target_in_self_goal_episodes":
            ci(rcpt_dist(ep, 0)[own]) if own.any() else None,
        "bystander_final_distance_to_senders_target_in_self_goal_episodes":
            ci(rcpt_dist(ep, -1)[own]) if own.any() else None,
        "bystander_displacement_in_self_goal_episodes":
            ci((_pick(ep.pos[:, -1], rcpt) - _pick(ep.pos[:, 0], rcpt)).norm(dim=-1)[own]) if own.any() else None,
        "recipient_initial_distance": ci(rcpt_dist(ep, 0)[v]),
        "recipient_final_distance": ci(rcpt_dist(ep, -1)[v]),
        "recipient_final_distance_ablated": ci(rcpt_dist(ablated, -1)[v]),
        "paired_increase_when_ablated": ci(increase[v]),
        "sender_gaze_alignment_cosine": ci(cos.mean(1)[v]),
        "sender_gaze_to_target_distance": ci((sgaze[:, 1:] - target.unsqueeze(1)).norm(dim=-1).mean(1)[v]),
        "sender_final_gaze_to_target_distance": ci((sgaze[:, -1] - target).norm(dim=-1)[v]),
        "sender_initial_distance_to_target": ci(to_target[:, 0].norm(dim=-1)[v]),
        "sender_final_distance_to_target": ci(to_target[:, -1].norm(dim=-1)[v]),
        "sender_closest_distance_to_target": ci(to_target.norm(dim=-1).amin(1)[v]),
        "share_of_episodes_with_contact":
            float((sep.amin(1) < 2 * cfg.agent_radius)[v].float().mean()) if cfg.collisions else None,
    }
    stiff = EnvConfig().contact_margin
    if cfg.collisions and cfg.contact_margin != stiff:
        # The policy was trained with a softer contact force. The same policy under the default force:
        hard_cfg = replace(cfg, contact_margin=stiff)
        hard = play(policy, hard_cfg, state, seed=1)
        hard_sep = (hard.pos[:, :, 0] - hard.pos[:, :, 1]).norm(dim=-1)
        out["default_contact_force"] = {
            "contact_margin_trained": cfg.contact_margin, "contact_margin_default": stiff,
            "recipient_final_distance": ci(rcpt_dist(hard, -1)[v]),
            "recipient_final_distance_ablated": ci(rcpt_dist(ablated, -1)[v]),
            "paired_increase_when_ablated": ci((rcpt_dist(ablated, -1) - rcpt_dist(hard, -1))[v]),
            "share_of_episodes_with_contact": float((hard_sep.amin(1) < 2 * cfg.agent_radius)[v].float().mean()),
        }
    idx = torch.nonzero(v)[:64, 0]
    arrays = {"sample_pos": ep.pos[idx].numpy(), "sample_gaze": ep.gaze[idx].numpy(),
              "sample_pos_ablated": ablated.pos[idx].numpy(), "sample_lm_pos": state.lm_pos[idx].numpy(),
              "sample_lm_color": state.lm_color[idx].numpy(), "sample_agent_color": state.agent_color[idx].numpy(),
              "sample_sender": sender[idx].numpy(), "sample_target": target[idx].numpy(),
              "paired_increase": increase[idx].numpy()}
    return out, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--baseline", type=Path, help="no-communication run trained on the same case")
    p.add_argument("--trials", type=int, default=10_000)
    p.add_argument("--probe-trials", type=int, default=4_000)
    p.add_argument("--out", type=Path)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    policy, tcfg = load(a.run_dir)
    out_dir = a.out or a.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not tcfg.env.comm:
        summary, arrays = nonverbal_eval(policy, tcfg, a.trials)
    elif not tcfg.comm:
        state = test_worlds(tcfg.env, a.trials)
        summary = {"case": tcfg.case, "trials": a.trials,
                   "reward": {"no_communication": reward_summary(policy, tcfg.env, state, comm=False)},
                   "centroid": centroid_stats(play(policy, tcfg.env, state, comm=False), state)}
        arrays = {}
    else:
        baseline = None
        if a.baseline:
            baseline, bcfg = load(a.baseline)
            if bcfg.comm or bcfg.case != tcfg.case or bcfg.env.horizon != tcfg.env.horizon:
                p.error(f"{a.baseline} is not a no-communication run of {tcfg.case} with horizon {tcfg.env.horizon}")
        init_policy = load(a.run_dir, "init")[0]
        summary, arrays = verbal_eval(policy, tcfg, a.trials, baseline, init_policy)
        summary["baseline"] = str(a.baseline) if a.baseline else None
        summary["baseline_seed"] = bcfg.seed if a.baseline else None
        summary["probe_trials"] = a.probe_trials
        labels = summary["symbols"]["labels"]
        summary["context_usage"] = context_usage(policy, tcfg, a.probe_trials, labels)
        if tcfg.env.n_agents == 3:
            summary["generalization"] = {
                "extra_agent": probe_extra_agent(policy, tcfg.env, a.probe_trials),
                "conflicting_goals": probe_conflict(policy, tcfg.env, a.probe_trials),
                "extra_landmark": probe_landmarks(policy, tcfg.env, a.probe_trials),
            }
    summary["seed"] = tcfg.seed
    metrics = np.load(a.run_dir / "metrics.npz")
    main_ctx = metrics["ctx"] == 0
    summary["train_reward_last_200_iters"] = float(metrics["reward"][main_ctx][-200:].mean())
    summary["train_terminal_last_200_iters"] = float(metrics["terminal"][main_ctx][-200:].mean())
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=1))
    if arrays:
        np.savez_compressed(out_dir / "eval_episodes.npz", **arrays)
    print(json.dumps({key: summary[key] for key in ("case", "seed", "train_reward_last_200_iters")}))


if __name__ == "__main__":
    main()
