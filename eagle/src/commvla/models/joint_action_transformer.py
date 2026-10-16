"""Small shared Transformer for left/right action-intent interaction."""

from __future__ import annotations

import torch
import torch.nn as nn


class JointActionTokenBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.0, batch_first=True)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        mlp_hidden = hidden_size * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        normalized = self.attn_norm(tokens)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.mlp_norm(tokens))


class JointActionTokenTransformer(nn.Module):
    """Jointly update one left and one right action token.

    The final projection is zero initialized, so the module starts as an exact
    local-token identity and gradually learns a cross-arm residual.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        num_layers: int = 3,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        device: str | int = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        if num_layers < 1 or mlp_ratio < 1:
            raise ValueError("num_layers and mlp_ratio must be positive")
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = int(mlp_ratio)
        self.side_embedding = nn.Parameter(torch.zeros(2, hidden_size))
        self.blocks = nn.ModuleList(
            JointActionTokenBlock(hidden_size, num_heads, mlp_ratio) for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.to(device=device, dtype=dtype)
        nn.init.normal_(self.side_embedding, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        left_token: torch.Tensor,
        right_token: torch.Tensor,
        *,
        allow_cross_arm: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if left_token.shape != right_token.shape:
            raise ValueError(f"left/right token shapes differ: {left_token.shape} vs {right_token.shape}")
        original_shape = left_token.shape
        if left_token.ndim == 1:
            left_flat = left_token.unsqueeze(0)
            right_flat = right_token.unsqueeze(0)
        elif left_token.ndim == 2:
            left_flat = left_token
            right_flat = right_token
        elif left_token.ndim == 3 and left_token.shape[1] == 1:
            left_flat = left_token[:, 0]
            right_flat = right_token[:, 0]
        else:
            raise ValueError(f"expected [D], [B,D], or [B,1,D] action tokens, got {left_token.shape}")
        original = torch.stack([left_flat, right_flat.to(left_flat.device)], dim=1)
        tokens = original + self.side_embedding.to(device=original.device, dtype=original.dtype).unsqueeze(0)
        attention_mask = None
        if not allow_cross_arm:
            attention_mask = torch.tensor(
                [[False, True], [True, False]], device=original.device, dtype=torch.bool
            )
        for block in self.blocks:
            tokens = block(tokens, attention_mask)
        correction = self.output_projection(self.output_norm(tokens))
        fused = original + correction
        fused_left = fused[:, 0]
        fused_right = fused[:, 1].to(right_token.device)
        if len(original_shape) == 1:
            return fused_left[0], fused_right[0]
        if len(original_shape) == 3:
            return fused_left.unsqueeze(1), fused_right.unsqueeze(1)
        return fused_left, fused_right
