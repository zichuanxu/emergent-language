"""Shared recurrent policy of Figure 3.

FC_x encodes every observed entity, FC_c every utterance stream (with a private
memory per stream), both are softmax-pooled, and FC_a maps the pooled features,
the private goal and the agent memory to movement, gaze, utterance logits and a
memory update. All modules have two hidden layers of ELU units with dropout.

The auxiliary head predicts each speaker's goal. It is not an input to FC_a.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .world import AUX_DIM, ENTITY_DIM, GOAL_DIM

SILENCE = 0


@dataclass(frozen=True)
class PolicyConfig:
    hidden: int = 256
    feature: int = 256
    memory: int = 32
    vocab: int = 20            # includes the silence symbol
    dropout: float = 0.1
    temperature: float = 1.0
    move_scale: float = 10.0
    action_noise: float = 0.1  # std of the Gaussian added to force and gaze
    memory_noise: float = 0.05
    aux_grounded: bool = True  # auxiliary head also reads the listener's pooled physical features
    logit_scale: float = 1.0   # multiplies the utterance logits (sharper initial utterances when > 1)
    goal_gain: float = 1.0     # multiplies the private goal vector at the input of FC_a
    init: str = "default"      # default (PyTorch Linear init) | he | xavier
    straight_through: bool = False  # train on one-hot utterances with the relaxed sample's gradient


def mlp(n_in: int, hidden: int, n_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.ELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ELU(),
        nn.Linear(hidden, n_out),
    )


def softmax_pool(h: Tensor, mask: Tensor | None = None) -> Tensor:
    """Feature-wise softmax-weighted sum over the set dimension (-2)."""
    if mask is None:
        return (h * h.softmax(-2)).sum(-2)
    mask = mask.unsqueeze(-1)
    weights = h.masked_fill(~mask, float("-inf")).softmax(-2)
    return (weights * h.masked_fill(~mask, 0.0)).sum(-2)


def gumbel_sample(logits: Tensor, temperature: float, hard: bool, straight_through: bool = False) -> Tensor:
    """Relaxed sample G(log p) = softmax((log p + eps) / tau), or an exact one-hot sample.

    With `straight_through` the relaxed sample is replaced by the one-hot sample it rounds to in the
    forward pass and keeps the relaxed sample's gradient (Jang et al. 2017). The paper trains on the
    relaxed sample itself; this is an option for comparison.
    """
    u = torch.rand_like(logits).clamp_(1e-10, 1 - 1e-7)
    y = logits - (-u.log()).log()
    if hard:
        return torch.zeros_like(logits).scatter_(-1, y.argmax(-1, keepdim=True), 1.0)
    soft = (y / temperature).softmax(-1)
    if straight_through:
        return torch.zeros_like(logits).scatter_(-1, y.argmax(-1, keepdim=True), 1.0) + soft - soft.detach()
    return soft


class Policy(nn.Module):
    def __init__(self, cfg: PolicyConfig = PolicyConfig()):
        super().__init__()
        self.cfg = cfg
        h, f, m, k = cfg.hidden, cfg.feature, cfg.memory, cfg.vocab
        self.fc_x = mlp(ENTITY_DIM, h, f, cfg.dropout)
        self.fc_c = mlp(k + m, h, f + m, cfg.dropout)
        self.fc_a = mlp(2 * f + GOAL_DIM + m, h, 4 + k + m, cfg.dropout)
        self.fc_g = nn.Sequential(
            nn.Linear(2 * f if cfg.aux_grounded else f, h), nn.ELU(), nn.Linear(h, AUX_DIM))
        for module in (self.fc_x, self.fc_c, self.fc_a, self.fc_g):
            layers = [layer for layer in module if isinstance(layer, nn.Linear)]
            for layer in layers if cfg.init == "xavier" else layers[:-1] if cfg.init == "he" else []:
                if cfg.init == "xavier":   # Glorot-uniform weights, zero biases (TensorFlow 1.x layer default)
                    nn.init.xavier_uniform_(layer.weight)
                else:                      # He-normal hidden layers, PyTorch default output layer
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def initial_memory(self, batch: int, n: int, device) -> tuple[Tensor, Tensor]:
        m = self.cfg.memory
        return torch.zeros(batch, n, n, m, device=device), torch.zeros(batch, n, m, device=device)

    def forward(self, ent, mask, goal, heard, memory, comm: bool = True):
        """heard: B,N,N,K — row i holds what listener i received on each stream.
        Returns move B,N,2 (force), gaze B,N,2, logits B,N,K, goal prediction B,N,N,AUX_DIM, memory."""
        cfg = self.cfg
        mem_c, mem_a = memory
        b, n = goal.shape[:2]
        phi_x = softmax_pool(self.fc_x(ent), mask)

        if comm:
            out = self.fc_c(torch.cat((heard, mem_c), -1))
            h_c, d_c = out[..., : cfg.feature], out[..., cfg.feature:]
            phi_c = softmax_pool(h_c)
            mem_c = self._update(mem_c, d_c)
            ctx = phi_x.unsqueeze(2).expand(-1, -1, n, -1)
            pred = self.fc_g(torch.cat((h_c, ctx), -1) if cfg.aux_grounded else h_c)
        else:
            phi_c = phi_x.new_zeros(b, n, cfg.feature)
            pred = phi_x.new_zeros(b, n, n, AUX_DIM)

        out = self.fc_a(torch.cat((phi_x, phi_c, goal * cfg.goal_gain, mem_a), -1))
        move, gaze = out[..., :2] * cfg.move_scale, out[..., 2:4]
        logits, d_a = out[..., 4: 4 + cfg.vocab] * cfg.logit_scale, out[..., 4 + cfg.vocab:]
        if cfg.action_noise:
            move = move + cfg.action_noise * torch.randn_like(move)
            gaze = gaze + cfg.action_noise * torch.randn_like(gaze)
        return move, gaze, logits, pred, (mem_c, self._update(mem_a, d_a))

    def _update(self, mem: Tensor, delta: Tensor) -> Tensor:
        noise = self.cfg.memory_noise * torch.randn_like(mem) if self.cfg.memory_noise else 0.0
        return (mem + delta + noise).tanh()
