"""Batched, differentiable particle world of Mordatch & Abbeel (2018).

What the paper fixes (Appendix): the Euler transition
    p' = p + v*dt,   v' = gamma*v + (u + f)*dt,   gaze' = u_gaze,
dt = 0.1, damping 1 - gamma = 0.5, and the private frame  i_p_j = R_i (p_j - p_i).

What the paper leaves open and this file decides (see README "Reconstruction
choices"): spawn area, reward formula, goal encoding, contact model.
"""

from dataclasses import dataclass, replace
from math import pi

import torch
from torch import Tensor
from torch.nn import functional as F

GOTO, LOOKAT, NOTHING, NOGOAL = 0, 1, 2, 3
ACTION_NAMES = ("GOTO", "LOOKAT", "DONOTHING")
COLOR_NAMES = ("RED", "GREEN", "BLUE")
ENTITY_DIM = 11  # rel pos 2, rel vel 2, rel gaze 2, rgb 3, is_agent 1, is_self 1
GOAL_DIM = 8     # action one-hot 3, target location in own frame 2, addressee rgb 3
AUX_DIM = 13     # goal as the listener would state it 8, landmark rgb 3, location in the speaker's frame 2
PALETTE = torch.eye(3)
NOVEL_COLOR = torch.tensor([1.0, 1.0, 0.0])


@dataclass(frozen=True)
class EnvConfig:
    n_agents: int = 2
    n_landmarks: int = 3
    actions: tuple = (GOTO,)
    goal_mode: str = "swap"      # swap | perm | oneway
    oneway_self: float = 0.0     # oneway: probability that the single goal addresses its own holder
    close_start: float = 0.0     # oneway: probability that the goal holder starts near the other agent
    comm: bool = True            # verbal channel available
    see_agents: bool = False     # other agents' position and velocity observed
    see_gaze: bool = False       # other agents' gaze location observed
    blind_recipient: bool = False  # one-way recipient observes no other agent
    collisions: bool = False
    horizon: int = 10
    dt: float = 0.1
    damping: float = 0.5
    arena: float = 1.0           # entities spawn uniformly in [-arena, arena]^2
    agent_radius: float = 0.1
    contact_force: float = 100.0
    contact_margin: float = 0.02
    close_max: float = 2.5       # close_start: largest starting centre distance, in body diameters
    self_actions: tuple = ()     # oneway: action types of self-goals when they differ from `actions`

    def __post_init__(self):
        assert self.goal_mode in ("swap", "perm", "oneway")
        assert 1 <= self.n_agents <= 3 and 1 <= self.n_landmarks <= 3
        if self.goal_mode in ("swap", "oneway"):
            assert self.n_agents == 2


@dataclass
class State:
    """World-frame tensors. B batch, N agents, M landmarks."""

    pos: Tensor          # B,N,2
    vel: Tensor          # B,N,2
    gaze: Tensor         # B,N,2  gaze location
    angle: Tensor        # B,N    private frame orientation
    agent_color: Tensor  # B,N,3
    lm_pos: Tensor       # B,M,2
    lm_color: Tensor     # B,M,3
    goal_type: Tensor    # B,N    GOTO | LOOKAT | NOTHING | NOGOAL
    goal_target: Tensor  # B,N,2  goal location (placed on a landmark)
    goal_lm: Tensor      # B,N    index of that landmark (bookkeeping only, never observed)
    goal_rcpt: Tensor    # B,N    index of the agent that should act
    blind: Tensor        # B,N    bool, cannot observe other agents
    home: Tensor         # B,N,2  position at the start of the episode

    def clone(self, **changes):
        return replace(self, **changes)


def rotation(angle: Tensor) -> Tensor:
    c, s = angle.cos(), angle.sin()
    return torch.stack((torch.stack((c, -s), -1), torch.stack((s, c), -1)), -2)


def _take(items: Tensor, index: Tensor) -> Tensor:
    return items.gather(1, index.unsqueeze(-1).expand(-1, -1, items.shape[-1]))


def sample(cfg: EnvConfig, batch: int, gen: torch.Generator | None = None, device="cpu") -> State:
    def rand(*shape):
        return torch.rand(batch, *shape, generator=gen).to(device)

    n, m = cfg.n_agents, cfg.n_landmarks
    pos = (2 * rand(n, 2) - 1) * cfg.arena
    lm_pos = (2 * rand(m, 2) - 1) * cfg.arena
    palette = PALETTE.to(device)
    agent_color = palette[rand(3).argsort(-1)[:, :n]]
    lm_color = palette[rand(3).argsort(-1)[:, :m]]
    actions = torch.tensor(cfg.actions, device=device)
    goal_type = actions[(rand(n) * len(actions)).long().clamp_max(len(actions) - 1)]
    goal_lm = (rand(n) * m).long().clamp_max(m - 1)
    blind = torch.zeros(batch, n, dtype=torch.bool, device=device)
    if cfg.goal_mode == "swap":
        goal_rcpt = torch.tensor([1, 0], device=device).expand(batch, 2)
    elif cfg.goal_mode == "perm":
        goal_rcpt = rand(n).argsort(-1)
    else:
        sender = (rand(1) < 0.5).long()
        own = rand(1) < cfg.oneway_self
        goal_rcpt = torch.where(own, torch.arange(2, device=device), torch.tensor([1, 0], device=device))
        if cfg.self_actions:
            own_actions = torch.tensor(cfg.self_actions, device=device)
            own_type = own_actions[(rand(1) * len(own_actions)).long().clamp_max(len(own_actions) - 1)]
            goal_type = torch.where(own, own_type, goal_type)
        goal_type = torch.where(torch.arange(2, device=device) == sender, goal_type, NOGOAL)
        if cfg.blind_recipient:
            blind = torch.arange(2, device=device) != sender
        if cfg.close_start:
            # Training aid for pushing: start the goal holder 1 to close_max body diameters from the other
            # agent, in a random direction that is independent of the goal.
            other = pos.gather(1, (1 - sender).unsqueeze(-1).expand(-1, -1, 2))
            angle = 2 * pi * rand(1, 1)
            dist = 2 * cfg.agent_radius * (1.0 + (cfg.close_max - 1.0) * rand(1, 1))
            near = other + dist * torch.cat((angle.cos(), angle.sin()), -1)
            move = (rand(1, 1) < cfg.close_start) & (torch.arange(2, device=device) == sender).unsqueeze(-1)
            pos = torch.where(move, near, pos)
    return State(
        pos=pos, vel=torch.zeros_like(pos), gaze=pos.clone(), angle=2 * pi * rand(n),
        agent_color=agent_color, lm_pos=lm_pos, lm_color=lm_color,
        goal_type=goal_type, goal_target=_take(lm_pos, goal_lm), goal_lm=goal_lm,
        goal_rcpt=goal_rcpt, blind=blind, home=pos.clone(),
    )


def to_frame(rot: Tensor, origin: Tensor, points: Tensor) -> Tensor:
    """points B,J,2 seen from each of N agents -> B,N,J,2."""
    return torch.einsum("bnij,bnkj->bnki", rot, points.unsqueeze(1) - origin.unsqueeze(2))


def observe(cfg: EnvConfig, s: State):
    """Entity features B,N,E,ENTITY_DIM, visibility mask B,N,E and private goals B,N,GOAL_DIM.

    An agent always observes itself and the M landmarks. With see_agents the entity list is all
    N agents followed by the landmarks (a blind agent's view of the others is masked); otherwise
    it is the agent itself followed by the landmarks. Other agents' gaze needs see_gaze.
    """
    b, n = s.pos.shape[:2]
    m = s.lm_pos.shape[1]
    dev = s.pos.device
    rot = rotation(s.angle)
    own_vel = torch.einsum("bnij,bnj->bni", rot, s.vel)
    own_gaze = torch.einsum("bnij,bnj->bni", rot, s.gaze - s.pos)
    lm = torch.cat((to_frame(rot, s.pos, s.lm_pos), s.lm_pos.new_zeros(b, n, m, 4),
                    s.lm_color.unsqueeze(1).expand(-1, n, -1, -1), s.lm_pos.new_zeros(b, n, m, 2)), -1)
    if cfg.see_agents:
        eye = torch.eye(n, dtype=torch.bool, device=dev)
        gaze = to_frame(rot, s.pos, s.gaze)
        if not cfg.see_gaze:
            gaze = gaze * eye.unsqueeze(-1)
        agents = torch.cat((
            to_frame(rot, s.pos, s.pos), torch.einsum("bnij,bkj->bnki", rot, s.vel), gaze,
            s.agent_color.unsqueeze(1).expand(-1, n, -1, -1), s.pos.new_ones(b, n, n, 1),
            eye.float().expand(b, -1, -1).unsqueeze(-1)), -1)
        visible = eye | ~s.blind.unsqueeze(-1)
        agents = agents * visible.unsqueeze(-1)
    else:
        agents = torch.cat((s.pos.new_zeros(b, n, 2), own_vel, own_gaze, s.agent_color,
                            s.pos.new_ones(b, n, 2)), -1).unsqueeze(2)
        visible = torch.ones(b, n, 1, dtype=torch.bool, device=dev)
    ent = torch.cat((agents, lm), 2)
    mask = torch.cat((visible, torch.ones(b, n, m, dtype=torch.bool, device=dev)), -1)

    has_goal = (s.goal_type != NOGOAL).unsqueeze(-1)
    has_loc = (s.goal_type <= LOOKAT).unsqueeze(-1)
    action = F.one_hot(s.goal_type, 4)[..., :3].float()
    target = torch.einsum("bnij,bnj->bni", rot, s.goal_target - s.pos) * has_loc
    rcpt_color = _take(s.agent_color, s.goal_rcpt) * has_goal
    return ent, mask, torch.cat((action, target, rcpt_color), -1)


def peer_goal_targets(s: State) -> Tensor:
    """Auxiliary targets B,N,N,AUX_DIM: candidate descriptions of speaker j's goal for listener i.

    [0:3 action | 3:5 location in the listener's frame | 5:8 addressee colour |
     8:11 colour of the landmark at the location | 11:13 location in the speaker's frame]
    The training configuration selects which of these enter the auxiliary loss.
    """
    b, n = s.pos.shape[:2]
    rot = rotation(s.angle)
    has_goal = (s.goal_type != NOGOAL).unsqueeze(-1)
    has_loc = (s.goal_type <= LOOKAT).unsqueeze(-1)
    action = F.one_hot(s.goal_type, 4)[..., :3].float()
    loc_listener = to_frame(rot, s.pos, s.goal_target) * has_loc.unsqueeze(1)
    loc_speaker = torch.einsum("bnij,bnj->bni", rot, s.goal_target - s.pos) * has_loc
    rcpt_color = _take(s.agent_color, s.goal_rcpt) * has_goal
    lm_color = _take(s.lm_color, s.goal_lm) * has_loc
    wide = lambda x: x.unsqueeze(1).expand(-1, n, -1, -1)
    return torch.cat((wide(action), loc_listener, wide(rcpt_color), wide(lm_color), wide(loc_speaker)), -1)


def contact_forces(cfg: EnvConfig, pos: Tensor) -> Tensor:
    """Smooth repulsion between agent bodies (softplus penetration)."""
    delta = pos.unsqueeze(2) - pos.unsqueeze(1)
    dist = (delta.square().sum(-1) + 1e-8).sqrt()
    pen = cfg.contact_margin * F.softplus((2 * cfg.agent_radius - dist) / cfg.contact_margin)
    scale = cfg.contact_force * pen / dist
    scale = scale * (1 - torch.eye(pos.shape[1], device=pos.device))
    return (delta * scale.unsqueeze(-1)).sum(2)


def step(cfg: EnvConfig, s: State, move: Tensor, gaze: Tensor) -> State:
    """Appendix dynamics. move and gaze are given in each agent's private frame."""
    inv = rotation(s.angle).transpose(-1, -2)
    force = torch.einsum("bnij,bnj->bni", inv, move)
    if cfg.collisions:
        force = force + contact_forces(cfg, s.pos)
    return s.clone(
        pos=s.pos + cfg.dt * s.vel,
        vel=(1 - cfg.damping) * s.vel + cfg.dt * force,
        gaze=s.pos + torch.einsum("bnij,bnj->bni", inv, gaze),
    )


def goal_errors(s: State) -> Tensor:
    """Squared error of every goal, B,N (zero where no goal), all in squared world units:
    GOTO    - addressee's distance to the location,
    LOOKAT  - distance of the addressee's gaze point to the location,
    NOTHING - addressee's displacement from where it started the episode."""
    p = _take(s.pos, s.goal_rcpt)
    g = _take(s.gaze, s.goal_rcpt)
    h = _take(s.home, s.goal_rcpt)
    err = torch.stack(
        ((p - s.goal_target).square().sum(-1), (g - s.goal_target).square().sum(-1), (p - h).square().sum(-1),
         torch.zeros_like(p[..., 0])), -1)
    return err.gather(-1, s.goal_type.unsqueeze(-1)).squeeze(-1)


def reward(s: State) -> Tensor:
    """Shared physical reward B: negative squared goal error averaged over assigned goals."""
    count = (s.goal_type != NOGOAL).sum(-1).clamp_min(1)
    return -goal_errors(s).sum(-1) / count
