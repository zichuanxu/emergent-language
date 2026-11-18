"""Evaluation statistics, whole-rollout information flow, and training-loop behaviour."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

import emlang.evaluate as evaluate
from emlang.evaluate import (analyze_symbols, goal_factors, landmark_choice, landmark_layout, nonverbal_eval,
                             play, probe_conflict, silence_hook, test_worlds as held_out_worlds)
from emlang.policy import Policy, PolicyConfig
from emlang.train import CASES, Ensemble, TrainConfig, contexts_for, load, rollout, train
from emlang.world import GOTO, LOOKAT, NOTHING, _take, observe, sample

SMALL = PolicyConfig(hidden=24, feature=16, memory=6, dropout=0.0, action_noise=0.0, memory_noise=0.0)
NOISY = PolicyConfig(hidden=24, feature=16, memory=6)


def retarget(state, agent=0):
    """The same world with one speaker's goal moved to another landmark."""
    lm = state.goal_lm.clone()
    lm[:, agent] = (lm[:, agent] + 1) % state.lm_pos.shape[1]
    return state.clone(goal_lm=lm, goal_target=_take(state.lm_pos, lm))


def test_in_a_whole_rollout_a_goal_reaches_other_agents_only_through_messages():
    for case in ("1x1x3", "3x3x3"):
        cfg = CASES[case]
        torch.manual_seed(0)
        policy = Policy(replace(SMALL, logit_scale=8.0))
        state = held_out_worlds(cfg, 64, seed=3)
        changed = retarget(state)
        others = [i for i in range(cfg.n_agents) if i != 0]
        a, b = play(policy, cfg, state, seed=1), play(policy, cfg, changed, seed=1)
        assert not torch.equal(a.symbols[:, :, 0], b.symbols[:, :, 0])          # the speaker says something else
        assert not torch.equal(a.pos[:, :, others], b.pos[:, :, others])          # and listeners react
        for kwargs in ({"heard_hook": silence_hook}, {"comm": False}):
            a, b = play(policy, cfg, state, seed=1, **kwargs), play(policy, cfg, changed, seed=1, **kwargs)
            # the addressee of the changed goal may be agent 0 itself in 3x3x3; all other agents are blind to it
            assert torch.equal(a.pos[:, :, others], b.pos[:, :, others])
            assert torch.equal(a.gaze[:, :, others], b.gaze[:, :, others])


def test_utterances_are_heard_exactly_one_step_later_inside_rollout():
    cfg = CASES["1x1x3"]
    torch.manual_seed(0)
    policy = Policy(SMALL)
    state = held_out_worlds(cfg, 32, seed=5)
    seen = []
    ep = play(policy, cfg, state, seed=2, heard_hook=lambda t, s, heard: (seen.append(heard.clone()), heard)[1])
    assert len(seen) == cfg.horizon
    assert torch.equal(seen[0].argmax(-1), torch.zeros_like(seen[0].argmax(-1)))            # silence first
    for t in range(1, cfg.horizon):
        heard = seen[t]                                                                      # B,listener,speaker,K
        assert torch.equal(heard.sum(-1), torch.ones_like(heard.sum(-1)))                    # exact one-hot
        assert torch.equal(heard.argmax(-1), ep.symbols[:, t - 1].unsqueeze(1).expand(-1, cfg.n_agents, -1))


def test_auxiliary_head_never_influences_behaviour():
    cfg = CASES["1x2x3"]
    torch.manual_seed(0)
    policy = Policy(SMALL)
    state = sample(cfg, 8, torch.Generator().manual_seed(0))
    ep = rollout(policy, cfg, state, hard=False)
    behaviour = ep.reward.sum() + ep.probs.square().sum() + ep.final.pos.sum() + ep.final.gaze.sum()
    grads = torch.autograd.grad(behaviour, list(policy.fc_g.parameters()), allow_unused=True)
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in grads)
    aux = torch.autograd.grad(ep.aux_err.sum(), list(policy.fc_a.parameters()), allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0 for g in aux)     # the task does reach the speaker


def test_ensemble_replicas_stay_independent_with_dropout_and_noise():
    cfg = CASES["1x1x3"]
    torch.manual_seed(0)
    ens = Ensemble([Policy(NOISY) for _ in range(3)]).train()
    state = sample(cfg, 3 * 8, torch.Generator().manual_seed(0))
    ep = rollout(ens, cfg, state, hard=False)
    reward = ep.reward.reshape(3, 8, -1).mean((1, 2))
    grads = torch.autograd.grad(reward[1], list(ens.parameters()), allow_unused=True)
    norms = torch.stack([g.reshape(3, -1).norm(dim=1) for g in grads if g is not None]).sum(0)
    assert norms[1] > 0 and norms[0] == 0 and norms[2] == 0


def test_resuming_from_a_checkpoint_gives_the_same_run(tmp_path):
    kwargs = dict(policy=NOISY, batch=16, penalty_start=2, penalty_ramp=2, mixed=True, case="1x2x3",
                  env=CASES["1x2x3"], aux_frame="speaker")
    train(TrainConfig(iters=8, **kwargs), tmp_path / "full", threads=1, log_every=100)
    train(TrainConfig(iters=4, **kwargs), tmp_path / "split", threads=1, log_every=100)
    train(TrainConfig(iters=8, **kwargs), tmp_path / "split", threads=1, log_every=100)
    full, _ = load(tmp_path / "full")
    split, _ = load(tmp_path / "split")
    for p, q in zip(full.parameters(), split.parameters()):
        assert torch.equal(p, q)
    a, b = np.load(tmp_path / "full" / "metrics.npz"), np.load(tmp_path / "split" / "metrics.npz")
    assert np.array_equal(a["ctx"], b["ctx"]) and np.array_equal(a["reward"], b["reward"])
    import json
    assert json.loads((tmp_path / "split" / "config.json").read_text())["train"]["iters"] == 8
    init_full, init_split = (torch.load(tmp_path / d / "init.pt", weights_only=True)["model"] for d in ("full", "split"))
    assert all(torch.equal(init_full[k], init_split[k]) for k in init_full)      # init.pt is not overwritten on resume


def test_resuming_works_on_an_accelerator(tmp_path):
    """Generator states in a checkpoint must go back to the CPU even when the model is loaded onto a GPU."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
    if device is None:
        import pytest
        pytest.skip("no accelerator on this machine")
    kwargs = dict(policy=SMALL, batch=8, case="1x1x3", env=CASES["1x1x3"], aux_frame="speaker")
    train(TrainConfig(iters=2, **kwargs), tmp_path, device, threads=1, log_every=100)
    train(TrainConfig(iters=4, **kwargs), tmp_path, device, threads=1, log_every=100)
    assert len(np.load(tmp_path / "metrics.npz")["reward"]) == 4


def test_training_contexts_are_a_distribution_with_the_main_world_first():
    for case in ("1x1x3", "1x2x3", "3x3x3"):
        for mixed in (False, True):
            for share in (0.0, 0.3):
                contexts = contexts_for(case, CASES[case], mixed, share)
                assert contexts[0][0] == CASES[case]
                assert abs(sum(w for _, w in contexts) - 1.0) < 1e-9 and all(w > 0 for _, w in contexts)
                two = [w for c, w in contexts if c == replace(CASES[case], n_landmarks=2)]
                assert (two == [share]) if share else (two == [])


def synthetic_language(state, horizon=6):
    """Every speaker says ACTION then LANDMARK-COLOUR then ADDRESSEE-COLOUR, and nothing else."""
    f = goal_factors(state)
    b, n = state.goal_type.shape
    symbols = torch.zeros(b, horizon, n, dtype=torch.long)
    symbols[:, 0] = 1 + f["action"]                                              # 1..3
    symbols[:, 1] = torch.where(f["landmark"] >= 0, 4 + f["landmark"], 0)        # 4..6, silent for DO NOTHING
    symbols[:, 2] = 7 + f["addressee"]                                           # 7..9
    return symbols


def test_symbol_analysis_recovers_a_known_compositional_language():
    cfg = CASES["3x3x3"]
    state = held_out_worlds(cfg, 3000, seed=11)
    out = analyze_symbols(synthetic_language(state), state, 20, True)
    assert out["n_active"] == 9
    expected = {1: "GOTO", 2: "LOOKAT", 3: "DONOTHING", 4: "RED", 5: "GREEN", 6: "BLUE",
                7: "RED-AGENT", 8: "GREEN-AGENT", 9: "BLUE-AGENT"}
    assert {int(k): v for k, v in out["labels"].items()} == expected
    for name in ("action", "landmark", "addressee"):
        assert out["decode"][name]["accuracy"] == 1.0
        assert out["decode"][name]["majority_baseline"] < 0.4
    assert out["order"]["action_before_landmark"]["share"] == 1.0
    assert out["order"]["landmark_before_addressee"]["share"] == 1.0
    # a colour vocabulary says little about where the target sits relative to the others
    rel = out["decode"]["relative_to_RED"]
    assert abs(rel["accuracy"] - rel["accuracy_from_landmark_colour"]) < 0.02 and rel["accuracy"] < 0.75
    # streams attributed to the wrong speaker carry no information about the goal
    shuffled = analyze_symbols(synthetic_language(state).roll(1, dims=0), state, 20, True)
    assert shuffled["decode"]["landmark"]["accuracy"] < 0.4


def test_layout_conditioned_decoding_detects_reference_by_position():
    cfg = CASES["1x1x3"]
    state = held_out_worlds(cfg, 6000, seed=13)
    f = goal_factors(state)
    symbols = torch.zeros(6000, 6, 2, dtype=torch.long)
    symbols[:, 0] = 1 + f["relative_to_RED"]            # IS-RED / NEARER-TO-RED / FARTHER-FROM-RED
    out = analyze_symbols(symbols, state, 20, False)["decode"]
    assert out["relative_to_RED"]["accuracy"] == 1.0
    assert 0.6 < out["landmark"]["accuracy"] < 0.72     # words alone: red is known, the other two are a coin flip
    assert out["landmark"]["accuracy_given_landmark_layout"] > out["landmark"]["accuracy"] + 0.1
    assert out["landmark"]["accuracy_from_layout_alone"] < 0.4
    assert int(landmark_layout(state).max()) == 11 and int(landmark_layout(state).min()) == 0


class _Episode:
    def __init__(self, pos):
        self.pos = pos


def test_landmark_choice_counts_addressees_that_end_at_their_target():
    cfg = CASES["1x1x3"]
    state = held_out_worlds(cfg, 500, seed=17)
    on_target = torch.zeros(500, 2, 2)
    on_target[:, 1], on_target[:, 0] = state.goal_target[:, 0], state.goal_target[:, 1]   # addressee of goal i is 1 - i
    centroid = state.lm_pos.mean(1, keepdim=True).expand(-1, 2, -1)
    assert landmark_choice(_Episode(on_target.unsqueeze(1)), state)["addressee_ends_nearest_to_target_landmark"][0] == 1.0
    at_centroid = landmark_choice(_Episode(centroid.unsqueeze(1)), state)["addressee_ends_nearest_to_target_landmark"][0]
    assert 0.25 < at_centroid < 0.42


def test_probes_and_nonverbal_evaluation_run_and_use_ordinary_starts():
    torch.manual_seed(0)
    out = probe_conflict(Policy(SMALL), CASES["3x3x3"], 64)
    assert {"both", "only_first", "only_second", "both_messages_silenced"} <= set(out)
    assert "to_landmark_centroid" in out["both"] and "share_at_midpoint" in out["both"]
    for case in ("pointing", "guiding", "pushing"):
        cfg = replace(CASES[case], close_start=1.0, horizon=4)
        tcfg = TrainConfig(case=case, env=cfg, comm=False, policy=SMALL)
        stats, arrays = nonverbal_eval(Policy(SMALL), tcfg, 200)
        assert stats["directed_goto_episodes"] > 40 and stats["self_goal_final_distance"] is not None
        # with the training aid every sender would start within 0.5 of the other agent
        start = arrays["sample_pos"][:, 0]
        assert np.linalg.norm(start[:, 0] - start[:, 1], axis=-1).mean() > 0.8
        assert (stats["share_of_episodes_with_contact"] is None) == (case != "pushing")
        assert arrays["sample_pos"].shape[1] == 5


def test_gaze_of_others_is_hidden_unless_the_world_shows_it():
    for case, visible in (("guiding", False), ("pointing", True)):
        cfg = CASES[case]
        state = sample(cfg, 16, torch.Generator().manual_seed(0))
        state = state.clone(gaze=state.pos + 0.7)
        ent, _, _ = observe(cfg, state)
        other_gaze = torch.stack((ent[:, 0, 1, 4:6], ent[:, 1, 0, 4:6]), 1)
        assert bool(other_gaze.abs().sum() > 0) == visible


def test_decoding_survives_meaningless_filler_words():
    """A perfect colour word plus random other words: the colour must stay decodable, with or without layout."""
    cfg = CASES["1x1x3"]
    state = held_out_worlds(cfg, 6000, seed=19)
    f = goal_factors(state)
    gen = torch.Generator().manual_seed(0)
    symbols = torch.randint(4, 20, (6000, 6, 2), generator=gen)       # sixteen filler words
    symbols[:, 0] = 1 + f["landmark"]
    out = analyze_symbols(symbols, state, 20, False)
    assert out["n_active"] == 19
    decode = out["decode"]["landmark"]
    assert decode["accuracy"] > 0.98 and decode["accuracy_given_landmark_layout"] > 0.98
    assert decode["share_decoded_by_single_word_fallback"] > 0.5      # almost every word set is new
    assert {out["labels"][str(k)] for k in (1, 2, 3)} == {"RED", "GREEN", "BLUE"}
    assert all(out["labels"][str(k)] == f"S{k}" for k in range(4, 20))


def test_words_for_positions_are_labelled_and_fillers_are_not():
    cfg = CASES["1x1x3"]
    state = held_out_worlds(cfg, 4000, seed=23)
    f = goal_factors(state)
    symbols = torch.zeros(4000, 6, 2, dtype=torch.long)
    symbols[:, 0] = 1 + f["relative_to_GREEN"]
    symbols[:, 1] = 7                                                  # said in every stream: means nothing
    labels = analyze_symbols(symbols, state, 20, False)["labels"]
    assert labels == {"1": "GREEN", "2": "NEARER-TO-GREEN", "3": "FARTHER-FROM-GREEN", "7": "S7"}


def test_word_statistics_ignore_streams_about_the_speakers_own_goal():
    cfg = CASES["3x3x3"]
    state = held_out_worlds(cfg, 3000, seed=29)
    f = goal_factors(state)
    symbols = synthetic_language(state)
    own = f["self_goal"] == 1
    symbols[:, 0] = torch.where(own, 10, symbols[:, 0])                # a word used only for self-goals
    symbols[:, 1] = torch.where(own, 4, symbols[:, 1])                 # and RED said regardless of the goal
    out = analyze_symbols(symbols, state, 20, True)
    assert out["table"]["4"]["precision"] == 1.0 and out["labels"]["4"] == "RED"
    assert out["labels"]["10"] == "OWN-GOAL" and out["table"]["10"]["precision"] == 1.0
    assert out["table"]["4"]["share_of_its_streams_about_own_goal"] > 0.3     # said there too, but it still means RED
    assert out["decode"]["landmark"]["accuracy"] == 1.0


def test_landmark_choice_matches_a_loop_over_goals_with_three_agents():
    cfg = CASES["3x3x3"]
    state = held_out_worlds(cfg, 400, seed=31)
    gen = torch.Generator().manual_seed(1)
    final = state.lm_pos[torch.arange(400).unsqueeze(1), torch.randint(0, 3, (400, 3), generator=gen)]
    final = final + 0.01 * torch.randn(400, 3, 2, generator=gen)       # every agent ends on some landmark
    per_world = []
    for w in range(400):
        hits = []
        for i in range(3):
            rcpt = int(state.goal_rcpt[w, i])
            if int(state.goal_type[w, i]) != GOTO or rcpt == i:
                continue
            nearest = int((final[w, rcpt] - state.lm_pos[w]).norm(dim=-1).argmin())
            hits.append(float(nearest == int(state.goal_lm[w, i])))
        if hits:
            per_world.append(sum(hits) / len(hits))
    got = landmark_choice(_Episode(final.unsqueeze(1)), state)["addressee_ends_nearest_to_target_landmark"]
    assert got[3] == len(per_world) and abs(got[0] - sum(per_world) / len(per_world)) < 1e-6


def _fake_play(endpoint):
    """A stand-in for evaluate.play that puts agent 2 where `endpoint(state)` says."""
    def fake(policy, cfg, state, **kwargs):
        pos = state.pos.clone()
        pos[:, 2] = endpoint(state)
        return SimpleNamespace(pos=pos.unsqueeze(1))
    return fake


def test_conflict_probe_tells_averaging_from_obeying_one_command_and_from_the_centroid(monkeypatch):
    cfg = CASES["3x3x3"]
    midpoint = lambda s: (s.lm_pos[:, 0] + s.lm_pos[:, 1]) / 2
    centroid = lambda s: s.lm_pos.mean(1)
    one = lambda s: torch.where(torch.arange(len(s.pos)).unsqueeze(1) % 2 == 0, s.lm_pos[:, 0], s.lm_pos[:, 1])
    shares = {}
    for name, endpoint in (("midpoint", midpoint), ("centroid", centroid), ("one", one)):
        monkeypatch.setattr(evaluate, "play", _fake_play(endpoint))
        shares[name] = probe_conflict(None, cfg, 2000)["both"]
    assert shares["midpoint"]["share_at_midpoint"] == 1.0 and shares["midpoint"]["share_at_first_target"] == 0.0
    assert shares["midpoint"]["projection_along_segment"][0] == 0.5
    assert shares["centroid"]["share_at_landmark_centroid"] == 1.0 and shares["centroid"]["share_at_midpoint"] < 0.6
    assert shares["one"]["share_at_midpoint"] == 0.0 and shares["one"]["share_at_landmark_centroid"] < 0.5
    assert abs(shares["one"]["share_at_first_target"] - 0.5) < 0.01 and abs(shares["one"]["share_at_second_target"] - 0.5) < 0.01


def test_pointing_world_gives_look_at_goals_to_the_holder_and_go_to_goals_to_the_other():
    cfg = CASES["pointing"]
    state = sample(cfg, 4000, torch.Generator().manual_seed(0))
    has_goal = state.goal_type != 3
    assert bool((has_goal.sum(1) == 1).all())                          # one goal per world
    holder = has_goal.float().argmax(1)
    kind = state.goal_type[torch.arange(4000), holder]
    own = state.goal_rcpt[torch.arange(4000), holder] == holder
    assert bool((kind[own] == LOOKAT).all()) and bool((kind[~own] == GOTO).all())
    assert 0.45 < own.float().mean() < 0.55
    guiding = sample(CASES["guiding"], 2000, torch.Generator().manual_seed(0))
    assert bool((guiding.goal_type[guiding.goal_type != 3] == GOTO).all())


def test_nonverbal_statistics_on_a_scripted_pointing_episode(monkeypatch):
    """Sender walks sideways while looking straight at the target; the other agent goes to the target only when asked."""
    tcfg = TrainConfig(case="pointing", env=replace(CASES["pointing"], horizon=5), comm=False, policy=SMALL)

    def scripted(policy, cfg, state, **kwargs):
        b = len(state.pos)
        holder = (state.goal_type != 3).float().argmax(1)
        rows = torch.arange(b)
        target = state.goal_target[rows, holder]
        directed = state.goal_rcpt[rows, holder] != holder
        pos, gaze = [state.pos.clone()], [state.pos.clone()]
        for t in range(1, cfg.horizon + 1):
            now = pos[-1].clone()
            look = now.clone()
            # set from the position one step earlier: a short ray towards the target when instructing the
            # other agent, the target itself when the look-at goal is the holder's own
            here = now[rows, holder]
            ray = here + 0.3 * torch.nn.functional.normalize(target - here, dim=-1)
            look[rows, holder] = torch.where(directed.unsqueeze(1), ray, target)
            now[rows, holder] = here + torch.tensor([0.3, 0.0])
            frac = t / cfg.horizon
            other = now[rows, 1 - holder]
            now[rows, 1 - holder] = torch.where(directed.unsqueeze(1), other + (target - other) * frac / (1 - (t - 1) / cfg.horizon), other)
            pos.append(now), gaze.append(look)
        pos, gaze = torch.stack(pos, 1), torch.stack(gaze, 1)
        err = torch.zeros(b, cfg.horizon, 2)
        err[rows, :, holder] = (gaze[rows, 1:, holder] - target.unsqueeze(1)).square().sum(-1)
        return SimpleNamespace(pos=pos, gaze=gaze, goal_err=err, reward=-err.sum(-1))

    monkeypatch.setattr(evaluate, "play", scripted)
    stats, _ = nonverbal_eval(None, tcfg, 1000)
    assert stats["sender_gaze_alignment_cosine"][0] > 0.999            # with gaze[t] - pos[t] it would be lower
    assert stats["sender_final_gaze_to_target_distance"][0] > 0.3      # it points, it does not look at the spot
    assert stats["recipient_final_distance"][0] < 1e-4
    assert stats["self_goal_final_distance"][0] < 1e-4                 # the holder's gaze miss, not its position
    assert stats["bystander_displacement_in_self_goal_episodes"][0] < 1e-6
    by0 = stats["bystander_initial_distance_to_senders_target_in_self_goal_episodes"][0]
    assert abs(stats["bystander_final_distance_to_senders_target_in_self_goal_episodes"][0] - by0) < 1e-6 and by0 > 0.5


def test_decoding_is_scored_on_streams_it_was_not_fitted_on():
    """Words without any meaning: a decoder scored on its own fitting half would look better than chance."""
    cfg = CASES["1x1x3"]
    state = held_out_worlds(cfg, 240, seed=37)
    symbols = torch.randint(1, 20, (240, 6, 2), generator=torch.Generator().manual_seed(0))
    decode = analyze_symbols(symbols, state, 20, False)["decode"]["landmark"]
    assert decode["accuracy"] < 0.42 and decode["accuracy_given_landmark_layout"] < 0.42


def test_a_positional_code_among_filler_words_is_decoded_only_with_the_layout():
    """Every word set is new, so decoding rests on the word-by-word vote, which must be fitted per layout."""
    cfg = CASES["1x1x3"]
    state = held_out_worlds(cfg, 8000, seed=41)
    f = goal_factors(state)
    symbols = torch.randint(4, 20, (8000, 6, 2), generator=torch.Generator().manual_seed(0))
    symbols[:, 0] = 1 + f["relative_to_RED"]
    decode = analyze_symbols(symbols, state, 20, False)["decode"]["landmark"]
    assert decode["share_decoded_by_single_word_fallback_given_layout"] > 0.9
    assert 0.6 < decode["accuracy"] < 0.72 and decode["accuracy_given_landmark_layout"] > 0.97


def test_the_guiding_ablation_blinds_only_the_recipient_to_the_sender():
    tcfg = TrainConfig(case="guiding", env=replace(CASES["guiding"], horizon=4), comm=False, policy=SMALL)
    torch.manual_seed(0)
    policy = Policy(SMALL)
    stats, arrays = nonverbal_eval(policy, tcfg, 300)
    assert stats["ablation"] == "sender hidden from the recipient"
    # replay the ablated recipient by hand: its path must not depend on where the sender is
    cfg = replace(tcfg.env, close_start=0.0)
    state = held_out_worlds(cfg, 300)
    sender = (state.goal_type != 3).float().argmax(1)
    rows, rcpt = torch.arange(300), 1 - sender

    def hide(_t, s, ent, mask, goal):
        mask = mask.clone()
        mask[rows, rcpt, sender] = False
        return ent * mask.unsqueeze(-1), mask, goal

    moved = state.pos.clone()
    moved[rows, sender] = moved[rows, sender] + 0.5
    a = play(policy, cfg, state, seed=1, obs_hook=hide)
    b = play(policy, cfg, state.clone(pos=moved, home=moved), seed=1, obs_hook=hide)
    assert torch.equal(a.pos[rows, :, rcpt], b.pos[rows, :, rcpt])
    assert not torch.equal(a.pos[rows, :, sender] + 0.5, b.pos[rows, :, sender])        # the sender still sees the recipient
    directed = (state.goal_rcpt[rows, sender] == rcpt) & (state.goal_type[rows, sender] == GOTO)
    idx = torch.nonzero(directed)[:64, 0]
    assert np.array_equal(arrays["sample_pos_ablated"], a.pos[idx].numpy())
    # without the hook the recipient does see the sender
    plain = play(policy, cfg, state, seed=1)
    plain_moved = play(policy, cfg, state.clone(pos=moved, home=moved), seed=1)
    assert not torch.equal(plain.pos[rows, :, rcpt], plain_moved.pos[rows, :, rcpt])


def test_pushing_trained_with_soft_contact_is_also_scored_with_the_default_contact():
    env = replace(CASES["pushing"], horizon=4, contact_margin=0.1)
    tcfg = TrainConfig(case="pushing", env=env, comm=False, policy=SMALL)
    torch.manual_seed(0)
    policy = Policy(SMALL)
    stats, _ = nonverbal_eval(policy, tcfg, 300)
    stiff = stats["default_contact_force"]
    assert (stiff["contact_margin_trained"], stiff["contact_margin_default"]) == (0.1, 0.02)
    same_policy_default_world, _ = nonverbal_eval(policy, TrainConfig(case="pushing", env=replace(env, contact_margin=0.02),
                                                                    comm=False, policy=SMALL), 300)
    assert "default_contact_force" not in same_policy_default_world
    assert stiff["recipient_final_distance"] == same_policy_default_world["recipient_final_distance"]
    assert stiff["share_of_episodes_with_contact"] == same_policy_default_world["share_of_episodes_with_contact"]
    assert stiff["recipient_final_distance_ablated"] == stats["recipient_final_distance_ablated"]
    assert stiff["recipient_final_distance"] != stats["recipient_final_distance"]
