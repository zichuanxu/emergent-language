"""Policy structure, information flow and training objective."""
import math

import pytest
import torch

from emlang.policy import SILENCE, Policy, PolicyConfig, gumbel_sample, softmax_pool
from emlang.train import CASES, Ensemble, TrainConfig, rollout, train, vocabulary_reward, load, penalty_factor
from emlang.world import AUX_DIM, GOAL_DIM, observe, peer_goal_targets, rotation, sample

SMALL = PolicyConfig(hidden=24, feature=16, memory=6, dropout=0.0, action_noise=0.0, memory_noise=0.0)


def setup(case="1x1x3", batch=6, seed=0, policy_cfg=SMALL):
    torch.manual_seed(seed)
    cfg = CASES[case]
    return cfg, sample(cfg, batch, torch.Generator().manual_seed(seed)), Policy(policy_cfg)


def silence(batch, n, k=20):
    heard = torch.zeros(batch, n, n, k)
    heard[..., SILENCE] = 1
    return heard


def test_module_sizes_follow_the_paper():
    policy = Policy()
    for module in (policy.fc_x, policy.fc_c, policy.fc_a):
        linear = [m for m in module if isinstance(m, torch.nn.Linear)]
        assert [m.out_features for m in linear[:2]] == [256, 256]
        assert sum(isinstance(m, torch.nn.ELU) for m in module) == 2
        assert [m.p for m in module if isinstance(m, torch.nn.Dropout)] == [0.1]
    assert policy.cfg.feature == 256 and policy.cfg.memory == 32 and policy.cfg.vocab == 20
    assert policy.cfg.temperature == 1.0


def test_softmax_pool_is_permutation_invariant_and_ignores_masked_rows():
    h = torch.randn(4, 5, 7)
    perm = torch.randperm(5)
    assert torch.allclose(softmax_pool(h), softmax_pool(h[:, perm]), atol=1e-6)
    mask = torch.tensor([True, True, True, False, False]).expand(4, 5)
    assert torch.allclose(softmax_pool(h, mask), softmax_pool(h[:, :3]), atol=1e-6)
    assert torch.allclose(softmax_pool(h), (h * h.softmax(1)).sum(1))


def test_policy_handles_any_number_of_agents_and_landmarks():
    policy = Policy(SMALL)
    for case in ("1x1x3", "3x3x3", "pointing"):
        cfg = CASES[case]
        s = sample(cfg, 5, torch.Generator().manual_seed(1))
        n = s.pos.shape[1]
        move, gaze, logits, pred, _ = policy(*observe(cfg, s), silence(5, n), policy.initial_memory(5, n, "cpu"))
        assert move.shape == gaze.shape == (5, n, 2) and logits.shape == (5, n, 20)
        assert pred.shape == (5, n, n, AUX_DIM)


def test_gumbel_sample_matches_the_categorical_distribution():
    torch.manual_seed(0)
    logits = torch.tensor([2.0, 0.0, -1.0, 1.0]).expand(200_000, 4)
    hard = gumbel_sample(logits, 1.0, hard=True)
    assert ((hard == 0) | (hard == 1)).all() and (hard.sum(-1) == 1).all()
    assert torch.allclose(hard.mean(0), logits[0].softmax(0), atol=0.005)
    soft = gumbel_sample(logits, 1.0, hard=False)
    assert torch.allclose(soft.sum(-1), torch.ones(200_000), atol=1e-5) and (soft > 0).all()
    assert torch.allclose(torch.nn.functional.one_hot(soft.argmax(-1), 4).float().mean(0), logits[0].softmax(0), atol=0.005)


def test_goal_reaches_the_other_agent_only_through_its_utterance_one_step_later():
    cfg, s, policy = setup()
    ent, mask, goal = observe(cfg, s)
    goal = goal.clone().requires_grad_()
    memory = policy.initial_memory(6, 2, "cpu")
    move, _, logits, _, memory = policy(ent, mask, goal, silence(6, 2), memory)
    grad = torch.autograd.grad(move[:, 1].sum(), goal, retain_graph=True)[0]
    assert grad[:, 0].abs().sum() == 0 and grad[:, 1].abs().sum() > 0       # same step: no leak
    words = gumbel_sample(logits, 1.0, hard=False)
    for message, expect_flow in ((words, True), (words.detach(), False)):
        heard = message.unsqueeze(1).expand(6, 2, 2, 20)
        move2, *_ = policy(ent, mask, goal.detach(), heard, memory)
        grad = torch.autograd.grad(move2[:, 1].sum(), goal, retain_graph=True, allow_unused=True)[0]
        flow = 0.0 if grad is None else grad[:, 0].abs().sum().item()
        assert (flow > 0) == expect_flow


def test_no_channel_means_no_dependence_on_messages():
    cfg, s, policy = setup()
    obs = observe(cfg, s)
    memory = policy.initial_memory(6, 2, "cpu")
    a = policy(*obs, silence(6, 2), memory, comm=False)
    b = policy(*obs, torch.rand(6, 2, 2, 20), memory, comm=False)
    assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])


def test_rollout_is_differentiable_end_to_end_through_other_agents_utterances():
    """The speaker's parameters receive gradient from the listener's physical reward."""
    cfg, s, policy = setup(batch=16)
    ep = rollout(policy, cfg, s, hard=False)
    assert ep.reward.shape == (16, cfg.horizon) and ep.probs.shape == (16, cfg.horizon, 2, 20)
    ep.reward.mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.fc_c.parameters())
    assert policy.fc_c[0].weight.grad.abs().sum() > 0
    hard = rollout(policy, cfg, s, hard=True)
    assert set(hard.symbols.unique().tolist()) <= set(range(20))


def test_auxiliary_target_is_the_speakers_goal_in_the_listeners_frame():
    cfg, s, _ = setup("3x3x3", 32)
    target = peer_goal_targets(s)
    i, j = 0, 2
    expected = torch.einsum("bij,bj->bi", rotation(s.angle[:, i]), s.goal_target[:, j] - s.pos[:, i])
    spatial = s.goal_type[:, j] <= 1
    assert torch.allclose(target[:, i, j, 3:5][spatial], expected[spatial], atol=1e-5)
    assert torch.equal(target[:, i, j, :3].argmax(-1)[spatial], s.goal_type[:, j][spatial])
    own = observe(cfg, s)[2]
    assert torch.allclose(target[:, j, j, :8], own[:, j], atol=1e-5)   # a speaker's own-frame view is its goal vector
    assert torch.allclose(target[:, i, j, 11:13], own[:, j, 3:5], atol=1e-5)  # literal location for every listener
    colour = s.lm_color.gather(1, s.goal_lm.unsqueeze(-1).expand(-1, -1, 3))[:, j]
    assert torch.allclose(target[:, i, j, 8:11][spatial], colour[spatial])


def test_auxiliary_reward_is_zero_for_perfect_predictions_and_ignores_self():
    cfg, s, policy = setup("3x3x3", 8)
    ep = rollout(policy, cfg, s, hard=False)
    assert ep.aux_err.shape == (8, AUX_DIM) and (ep.aux("listener") < 0).all()
    assert torch.allclose(ep.aux("both"), ep.aux("listener") + ep.aux("speaker") - ep.aux("none"), atol=1e-5)
    assert rollout(policy, cfg, s, hard=False, comm=False).aux_err.abs().sum() == 0


def test_vocabulary_reward_matches_the_dirichlet_process_formula():
    probs = torch.zeros(4, 5, 2, 6)
    probs[..., 0] = 0.5
    probs[..., 1] = 0.3
    probs[..., 2] = 0.2
    slots = 4 * 5 * 2
    n1, n2 = 0.3 * slots, 0.2 * slots
    alpha = 1.0
    want = (n1 * math.log(n1 / (alpha + n1 + n2 - 1)) + n2 * math.log(n2 / (alpha + n1 + n2 - 1))) / slots
    reward, spoken = vocabulary_reward(probs, alpha)
    assert reward.item() == pytest.approx(want, rel=1e-5) and spoken.item() == pytest.approx(0.5)
    # consolidating the same amount of speech on one symbol is rewarded (rich get richer)
    merged = torch.zeros_like(probs)
    merged[..., 0], merged[..., 1] = 0.5, 0.5
    assert vocabulary_reward(merged, alpha)[0] > reward
    # silence is free
    quiet = torch.zeros_like(probs)
    quiet[..., 0] = 1.0
    assert vocabulary_reward(quiet, alpha)[0].item() == pytest.approx(0.0, abs=1e-6)


def test_ensemble_replicas_equal_separate_policies_and_do_not_interact():
    cfg = CASES["3x3x3"]
    torch.manual_seed(0)
    singles = [Policy(SMALL) for _ in range(3)]
    ensemble = Ensemble(singles)
    s = sample(cfg, 3 * 5, torch.Generator().manual_seed(3))
    obs = observe(cfg, s)
    heard = torch.rand(15, 3, 3, 20)
    memory = (torch.randn(15, 3, 3, 6), torch.randn(15, 3, 6))
    joint = ensemble(*obs, heard, memory)
    for r, policy in enumerate(singles):
        rows = slice(5 * r, 5 * r + 5)
        alone = policy(*(x[rows] for x in obs), heard[rows], (memory[0][rows], memory[1][rows]))
        for a, b in zip(joint[:4], alone[:4]):
            assert torch.allclose(a[rows], b, atol=1e-5)
        assert torch.allclose(joint[4][0][rows], alone[4][0], atol=1e-5)
    # the loss of replica 0 leaves the other replicas' parameters without gradient
    ep = rollout(ensemble, cfg, s, hard=False)
    (ep.reward[:5].mean() + ep.aux("both")[:5].mean()).backward()
    for p in ensemble.stacked:
        assert p.grad[0].abs().sum() > 0 and p.grad[1:].abs().sum() == 0
    norm = ensemble.clip_grad_norm_(torch.tensor([1e-3, 0.0, 0.0]))
    total = sum(p.grad[0].square().sum() for p in ensemble.stacked).sqrt()
    assert norm[0] > 1e-3 and total.item() == pytest.approx(1e-3, rel=1e-3)
    for key, value in ensemble.replica_state(1).items():
        assert torch.equal(value, dict(singles[1].named_parameters())[key])


def test_vocabulary_reward_per_replica():
    probs = torch.rand(3, 4, 5, 2, 6).softmax(-1)
    joint, spoken = vocabulary_reward(probs, torch.tensor([1.0, 1.0, 2.0]))
    for r, alpha in enumerate((1.0, 1.0, 2.0)):
        alone, s = vocabulary_reward(probs[r], alpha)
        assert joint[r].item() == pytest.approx(alone.item(), rel=1e-5) and spoken[r].item() == pytest.approx(s.item())


def test_penalty_schedule():
    cfg = TrainConfig(penalty_start=100, penalty_ramp=50)
    assert [penalty_factor(cfg, it) for it in (1, 100, 125, 150, 9999)] == [0.0, 0.0, 0.5, 1.0, 1.0]


def test_training_runs_checkpoints_and_resumes_identically(tmp_path):
    cfg = TrainConfig(policy=SMALL, iters=6, batch=16, penalty_start=2, penalty_ramp=2)
    train(cfg, tmp_path / "full", threads=1, log_every=100)
    policy, loaded = load(tmp_path / "full")
    assert loaded == cfg and not policy.training
    import numpy as np
    metrics = np.load(tmp_path / "full" / "metrics.npz")
    assert metrics["freq"].shape == (6, 20) and np.allclose(metrics["freq"].sum(1), 1.0, atol=1e-5)
    assert np.isfinite(metrics["reward"]).all()
    # two replicas in one process write one ordinary run directory each
    cfgs = [cfg, TrainConfig(policy=SMALL, iters=6, batch=16, penalty_start=2, penalty_ramp=2, seed=5, w_goal=0.1,
                             aux_frame="speaker")]
    train(cfgs, tmp_path / "pair", threads=1, log_every=100, names=["a", "b"])
    a, cfg_a = load(tmp_path / "pair" / "a")
    b, cfg_b = load(tmp_path / "pair" / "b")
    assert cfg_a == cfg and cfg_b.seed == 5 and cfg_b.aux_frame == "speaker"
    init_a, _ = load(tmp_path / "full", "init")
    init_pair, _ = load(tmp_path / "pair" / "a", "init")
    for p, q in zip(init_a.parameters(), init_pair.parameters()):
        assert torch.equal(p, q)                      # same seed, same initial weights
    assert any(not torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert np.load(tmp_path / "pair" / "b" / "metrics.npz")["freq"].shape == (6, 20)


def test_straight_through_sample_is_one_hot_and_carries_the_relaxed_gradient():
    torch.manual_seed(0)
    logits = torch.randn(64, 20, requires_grad=True)
    torch.manual_seed(1)
    y = gumbel_sample(logits, 1.0, hard=False, straight_through=True)
    torch.manual_seed(1)
    soft = gumbel_sample(logits, 1.0, hard=False)
    assert torch.equal((y == y.amax(-1, keepdim=True)).float(), y.detach().round())
    assert torch.allclose(y.sum(-1), torch.ones(64)) and torch.equal(y.argmax(-1), soft.argmax(-1))
    w = torch.randn(20)
    g1, = torch.autograd.grad((y * w).sum(), logits)
    g2, = torch.autograd.grad((soft * w).sum(), logits)
    assert torch.allclose(g1, g2, atol=1e-6) and g1.abs().sum() > 0
