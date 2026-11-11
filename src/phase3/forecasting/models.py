from typing import Dict, Iterable, List, Literal, Optional, Tuple

import torch
import torch.nn as nn


class EdgeLSTMForecaster(nn.Module):
    """
    Lightweight PyTorch LSTM workload forecaster for decentralized edge nodes.
    Supports:
      - P1 (`p1_shared`): entire model state_dict shared via Gossip
      - P2 (`p2_local_head`, recommended): LSTM + fc_mid backbone gossiped; output `head` stays local
      - P3 (`p3_finetune`): entire model shared + post-merge local fine-tune steps
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.15,
        personalization: Literal["p1_shared", "p2_local_head", "p3_finetune"] = "p2_local_head",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.personalization = personalization

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        mid_dim = max(8, hidden_dim // 2)
        self.fc_mid = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Linear head avoids dead-ReLU trapping after gossip weight merges
        self.head = nn.Linear(mid_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_h = out[:, -1, :]
        h = self.fc_mid(last_h)
        return self.head(h).squeeze(-1)

    @torch.no_grad()
    def predict_nonneg(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        preds = torch.clamp(self.forward(x), min=0.0, max=2.0)
        if was_training:
            self.train()
        return preds

    def shared_keys(self) -> List[str]:
        keys = list(self.state_dict().keys())
        if self.personalization == "p2_local_head":
            return [k for k in keys if not k.startswith("head.")]
        return keys

    def export_shared_weights(self) -> Dict[str, torch.Tensor]:
        sd = self.state_dict()
        return {k: sd[k].detach().cpu().clone() for k in self.shared_keys()}

    def load_shared_weights(self, shared: Dict[str, torch.Tensor]) -> None:
        sd = self.state_dict()
        for k, v in shared.items():
            if k in sd:
                sd[k].copy_(v.to(sd[k].device, dtype=sd[k].dtype))

    def shared_num_params(self) -> int:
        sd = self.state_dict()
        return sum(int(sd[k].numel()) for k in self.shared_keys())

    def total_num_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def serialized_payload_bytes(self, header_bytes: int = 24) -> int:
        sd = self.state_dict()
        tensor_bytes = sum(int(sd[k].numel() * sd[k].element_size()) for k in self.shared_keys())
        return tensor_bytes + header_bytes


def age_weighted_merge(
    local_weights: Dict[str, torch.Tensor],
    local_age: float,
    peer_packages: Iterable[Tuple[Dict[str, torch.Tensor], float]],
    age_decay: float = 0.05,
    self_mix_alpha: Optional[float] = None,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Bug-free age-weighted averaging (fixing Phase 2 B8 zero-age asymmetry & monotonic unbounded growth).
    Uses positive floor weights, exponential recency-capped age update, and optional self-anchor
    damping (`self_mix_alpha` in (0, 1)) to prevent multi-peer overwrites on heterogeneous nodes.
    """
    peer_list = [(w_dict, max(1.0, float(p_age))) for w_dict, p_age in peer_packages]
    if not peer_list:
        return {k: v.clone() for k, v in local_weights.items()}, max(1.0, float(local_age))

    keys = local_weights.keys()
    merged: Dict[str, torch.Tensor] = {}

    if self_mix_alpha is not None and 0.0 < self_mix_alpha < 1.0:
        peer_weight_sum = sum(w for _, w in peer_list)
        for k in keys:
            peer_acc = torch.zeros_like(local_weights[k], dtype=torch.float32)
            for w_dict, w_val in peer_list:
                peer_acc.add_(w_dict[k].to(dtype=torch.float32), alpha=float(w_val) / peer_weight_sum)
            blended = (
                float(self_mix_alpha) * local_weights[k].to(dtype=torch.float32)
                + (1.0 - float(self_mix_alpha)) * peer_acc
            )
            merged[k] = blended.to(dtype=local_weights[k].dtype)
        max_age = max(max(1.0, float(local_age)), max(w for _, w in peer_list))
    else:
        pkgs = [(local_weights, max(1.0, float(local_age)))] + peer_list
        weight_sum = sum(w for _, w in pkgs)
        for k in keys:
            acc = torch.zeros_like(local_weights[k], dtype=torch.float32)
            for w_dict, w_val in pkgs:
                acc.add_(w_dict[k].to(dtype=torch.float32), alpha=float(w_val) / weight_sum)
            merged[k] = acc.to(dtype=local_weights[k].dtype)
        max_age = max(w for _, w in pkgs)

    # Freshness-bounded merged sample/experience age (prevents stale nodes from permanently dominating late online phase)
    new_age = max(1.0, max_age * (1.0 - age_decay))
    return merged, float(new_age)

