"""World dynamics, private frames, goals and rewards."""
import math

import pytest
import torch

from emlang.world import (GOTO, LOOKAT, NOGOAL, NOTHING, EnvConfig, contact_forces, goal_errors, observe,
                          peer_goal_targets, reward, rotation, sample, step)
from emlang.train import CASES


def world(case="3x3x3", batch=64, seed=0):
    cfg = CASES[case]
    return cfg, sample(cfg, batch, torch.Generator().manual_seed(seed))


def test_appendix_euler_transition():
    cfg, s = world("1x1x3", 8)
    s = s.clone(vel=torch.randn(8, 2, 2), angle=torch.zeros(8, 2))
    force = torch.randn(8, 2, 2)
    nxt = step(cfg, s, force, torch.zeros(8, 2, 2))
    assert torch.allclose(nxt.pos, s.pos + 0.1 * s.vel)
    assert torch.allclose(nxt.vel, 0.5 * s.vel + 0.1 * force)


def test_force_and_gaze_are_applied_in_the_private_frame():
    cfg, s = world("1x1x3", 8)
    s = s.clone(angle=torch.full((8, 2), math.pi / 2))
    local = torch.tensor([1.0, 0.0]).expand(8, 2, 2)
    nxt = step(cfg, s, local, local)
    world_dir = torch.einsum("bnij,bnj->bni", rotation(s.angle).transpose(-1, -2), local)
    assert torch.allclose(nxt.vel, 0.1 * world_dir, atol=1e-6)
    assert torch.allclose(nxt.gaze - s.pos, world_dir, atol=1e-6)
    # observing the commanded gaze point returns the commanded local vector
    ent, _, _ = observe(cfg, nxt.clone(pos=s.pos))
    assert torch.allclose(ent[:, :, 0, 4:6], local, atol=1e-5)


def test_observation_is_invariant_to_a_global_rigid_motion():
    cfg, s = world("3x3x3")
    theta, shift = 0.7, torch.tensor([0.3, -1.1])
    rot = rotation(torch.tensor(theta))
    move = lambda p: p @ rot.T + shift
    moved = s.clone(pos=move(s.pos), gaze=move(s.gaze), lm_pos=move(s.lm_pos), goal_target=move(s.goal_target),
                    home=move(s.home),
                    vel=s.vel @ rot.T, angle=s.angle - theta)
    for a, b in zip(observe(cfg, s), observe(cfg, moved)):
        assert torch.allclose(a.float(), b.float(), atol=1e-5)
    assert torch.allclose(peer_goal_targets(s), peer_goal_targets(moved), atol=1e-5)


def test_private_frames_hide_the_shared_frame():
    """Rotating one agent's frame changes only that agent's observations."""
    cfg, s = world("3x3x3")
    turned = s.clone(angle=s.angle + torch.tensor([1.0, 0.0, 0.0]))
    a, b = observe(cfg, s), observe(cfg, turned)
    assert not torch.allclose(a[0][:, 0], b[0][:, 0])
    assert torch.allclose(a[0][:, 1:], b[0][:, 1:]) and torch.allclose(a[2][:, 1:], b[2][:, 1:])


def test_verbal_worlds_do_not_expose_other_agents():
    cfg, s = world("3x3x3")
    ent, mask, goal = observe(cfg, s)
    assert ent.shape[2] == 1 + cfg.n_landmarks and mask.all()
    far = s.clone(pos=s.pos + torch.tensor([[0.0, 0], [5, 5], [0, 0]]), gaze=s.gaze + torch.tensor([[0.0, 0], [5, 5], [0, 0]]))
    ent2, _, goal2 = observe(cfg, far)
    assert torch.allclose(ent[:, 0], ent2[:, 0]) and torch.allclose(ent[:, 2], ent2[:, 2])
    assert torch.allclose(goal[:, 0], goal2[:, 0])


def test_goal_vector_contains_location_but_no_landmark_identity():
    cfg, s = world("3x3x3", 256)
    _, _, goal = observe(cfg, s)
    spatial = s.goal_type <= LOOKAT
    local = torch.einsum("bnij,bnj->bni", rotation(s.angle), s.goal_target - s.pos)
    assert torch.allclose(goal[..., 3:5][spatial], local[spatial], atol=1e-5)
    assert (goal[..., 3:5][~spatial] == 0).all()
    assert torch.equal(goal[..., :3].argmax(-1), s.goal_type)
    assert goal.shape[-1] == 8  # action 3 + location 2 + addressee colour 3


def test_goals_never_conflict_and_cover_all_cases():
    cfg, s = world("3x3x3", 4096)
    assert (s.goal_rcpt.sort(-1).values == torch.arange(3)).all()
    assert set(s.goal_type.unique().tolist()) == {GOTO, LOOKAT, NOTHING}
    assert (s.agent_color.sum(1) == 1).all() and (s.lm_color.sum(1) == 1).all()
    self_share = (s.goal_rcpt == torch.arange(3)).float().mean().item()
    assert 0.28 < self_share < 0.39
    cfg2, s2 = world("1x1x3", 512)
    assert (s2.goal_rcpt == torch.tensor([1, 0])).all() and (s2.goal_type == GOTO).all()
    assert torch.allclose(s2.goal_target, s2.lm_pos.gather(1, s2.goal_lm.unsqueeze(-1).expand(-1, -1, 2)))


def test_reward_is_negative_squared_error_of_the_addressee():
    cfg, s = world("3x3x3", 16)
    s = s.clone(goal_type=torch.tensor([GOTO, LOOKAT, NOTHING]).expand(16, 3).clone(),
                goal_rcpt=torch.tensor([1, 2, 0]).expand(16, 3).clone(), vel=torch.randn(16, 3, 2),
                home=s.pos + 0.3 * torch.randn(16, 3, 2))
    err = goal_errors(s)
    assert torch.allclose(err[:, 0], (s.pos[:, 1] - s.goal_target[:, 0]).square().sum(-1))
    assert torch.allclose(err[:, 1], (s.gaze[:, 2] - s.goal_target[:, 1]).square().sum(-1))
    assert torch.allclose(err[:, 2], (s.pos[:, 0] - s.home[:, 0]).square().sum(-1))
    assert torch.allclose(reward(s), -err.mean(-1))
    solved = s.clone(pos=s.pos.clone().index_copy(1, torch.tensor([1]), s.goal_target[:, :1]),
                     gaze=s.gaze.clone().index_copy(1, torch.tensor([2]), s.goal_target[:, 1:2]))
    solved = solved.clone(home=solved.home.clone().index_copy(1, torch.tensor([0]), solved.pos[:, :1]))
    assert torch.allclose(reward(solved), torch.zeros(16), atol=1e-6)


def test_one_way_worlds_have_one_goal_and_blind_recipient():
    cfg, s = world("pushing", 512)
    assert ((s.goal_type != NOGOAL).sum(-1) == 1).all()
    sender = (s.goal_type != NOGOAL).float().argmax(1)
    self_share = (s.goal_rcpt[torch.arange(512), sender] == sender).float().mean().item()
    assert 0.4 < self_share < 0.6
    assert torch.equal(s.blind.float().argmax(1), 1 - sender)
    ent, mask, goal = observe(cfg, s)
    idx = torch.arange(512)
    assert not mask[idx, 1 - sender, sender].any() and mask[idx, sender, 1 - sender].all()
    assert (ent[idx, 1 - sender, sender] == 0).all()
    assert (goal[idx, 1 - sender] == 0).all()


def test_gaze_is_visible_only_when_pointing():
    for case, visible in (("pointing", True), ("guiding", False)):
        cfg, s = world(case, 32)
        s = s.clone(gaze=s.pos + 1.0)
        ent, _, _ = observe(cfg, s)
        other_gaze = ent[:, 0, 1, 4:6]
        assert (other_gaze.abs().sum(-1) > 0).all() == visible


def test_contact_force_is_repulsive_equal_and_opposite_and_local():
    cfg = CASES["pushing"]
    pos = torch.tensor([[[0.0, 0.0], [0.15, 0.0]], [[0.0, 0.0], [1.0, 0.0]]])
    f = contact_forces(cfg, pos)
    assert f[0, 0, 0] < 0 < f[0, 1, 0] and torch.allclose(f[0, 0], -f[0, 1])
    assert f[1].abs().max() < 1e-6


def test_centroid_is_the_best_blind_guess():
    """Analytic check of the no-communication optimum used to interpret Table 1."""
    gen = torch.Generator().manual_seed(0)
    lm = torch.rand(200_000, 3, 2, generator=gen) * 2 - 1
    err = (lm[:, 0] - lm.mean(1)).square().sum(-1).mean().item()
    assert err == pytest.approx(4 / 9, rel=0.02)


def test_close_start_places_the_goal_holder_near_but_not_on_the_other_agent():
    from dataclasses import replace
    cfg = replace(CASES["pushing"], close_start=1.0)
    s = sample(cfg, 2048, torch.Generator().manual_seed(0))
    gap = (s.pos[:, 0] - s.pos[:, 1]).norm(dim=-1)
    assert gap.min() >= 2 * cfg.agent_radius - 1e-5 and gap.max() <= 5 * cfg.agent_radius + 1e-5
    assert torch.equal(s.home, s.pos)
    sender = (s.goal_type != NOGOAL).float().argmax(1)
    target = s.goal_target[torch.arange(2048), sender]
    to_other = torch.nn.functional.normalize(s.pos[torch.arange(2048), 1 - sender] - s.pos[torch.arange(2048), sender], dim=-1)
    to_goal = torch.nn.functional.normalize(target - s.pos[torch.arange(2048), sender], dim=-1)
    assert abs((to_other * to_goal).sum(-1).mean().item()) < 0.3   # direction carries little about the goal
