from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SlotExtractor(nn.Module):
    """Lightweight CNN + MLP that extracts a slot vector and scalar depth."""

    def __init__(self, in_channels: int, slot_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, slot_dim + 1),
        )

    def forward(self, z_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(z_0).flatten(1)  # [B, 128]
        out = self.mlp(x)  # [B, slot_dim + 1]
        initial_slot = out[:, :-1]  # [B, slot_dim]
        slot_depth = out[:, -1:]  # [B, 1]
        return initial_slot, slot_depth


class SpatialBroadcastDecoder(nn.Module):
    """Decodes a slot state into a spatial latent map."""

    def __init__(self, slot_dim: int, out_channels: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(slot_dim + 2, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(self, slot: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, slot_dim = slot.shape
        slot_map = slot.view(batch_size, slot_dim, 1, 1).expand(-1, -1, height, width)

        y_coords = torch.linspace(-1.0, 1.0, steps=height, device=slot.device, dtype=slot.dtype)
        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=slot.device, dtype=slot.dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coord_grid = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)

        decoded = self.decoder(torch.cat([slot_map, coord_grid], dim=1))  # [B, C, H, W]
        return decoded


class DepthRoutedLatentWorldModel(nn.Module):
    """
    Inputs:
      - z_0: [B, C, H, W]
      - depth_map: [B, 1, H, W]
      - trajectory: [B, T, 2] in [-1, 1]

    Outputs:
      - z_hat: [B, T, C, H, W]
      - mask_hat: [B, T, 1, H, W]
    """

    def __init__(
        self,
        latent_channels: int,
        slot_dim: int = 128,
        use_tqdm: bool = False,
    ) -> None:
        super().__init__()
        self.slot_extractor = SlotExtractor(in_channels=latent_channels, slot_dim=slot_dim)
        self.physics_rnn = nn.GRUCell(input_size=2, hidden_size=slot_dim)
        self.renderer = SpatialBroadcastDecoder(slot_dim=slot_dim, out_channels=latent_channels)
        self.use_tqdm = use_tqdm

    @staticmethod
    def _route_depth(
        depth_map: torch.Tensor, trajectory_t: torch.Tensor, slot_depth: torch.Tensor
    ) -> torch.Tensor:
        """
        depth_map: [B, 1, H, W]
        trajectory_t: [B, 2]
        slot_depth: [B, 1]
        Returns visible mask scalar per sample: [B, 1], float in {0, 1}
        """
        grid = trajectory_t.view(trajectory_t.shape[0], 1, 1, 2)  # [B, 1, 1, 2]
        bg_depth = F.grid_sample(
            depth_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(trajectory_t.shape[0], 1)  # [B, 1]

        # Visible if object depth is in front of (or equal to) sampled background depth.
        visible = (slot_depth <= bg_depth).float()  # [B, 1]
        return visible

    def forward(
        self, z_0: torch.Tensor, depth_map: torch.Tensor, trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if z_0.ndim != 4:
            raise ValueError(f"z_0 must be [B, C, H, W], got shape {tuple(z_0.shape)}")
        if depth_map.ndim != 4:
            raise ValueError(f"depth_map must be [B, 1, H, W], got shape {tuple(depth_map.shape)}")
        if trajectory.ndim != 3 or trajectory.shape[-1] != 2:
            raise ValueError(f"trajectory must be [B, T, 2], got shape {tuple(trajectory.shape)}")

        batch_size, channels, height, width = z_0.shape
        _, time_steps, _ = trajectory.shape

        initial_slot, slot_depth = self.slot_extractor(z_0)  # [B, slot_dim], [B, 1]
        hidden = initial_slot

        z_seq = []
        m_seq = []

        time_iterator = range(time_steps)
        if self.use_tqdm:
            time_iterator = tqdm(time_iterator, desc="Model rollout", leave=False)

        for t in time_iterator:
            traj_t = trajectory[:, t, :]  # [B, 2]
            hidden = self.physics_rnn(traj_t, hidden)  # [B, slot_dim]

            visible_scalar = self._route_depth(depth_map, traj_t, slot_depth)  # [B, 1]
            mask_t = visible_scalar.view(batch_size, 1, 1, 1).expand(-1, 1, height, width)  # [B,1,H,W]

            rendered_t = self.renderer(hidden, height, width)  # [B, C, H, W]
            routed_t = rendered_t * mask_t  # [B, C, H, W]

            z_seq.append(routed_t)
            m_seq.append(mask_t)

        z_hat = torch.stack(z_seq, dim=1)  # [B, T, C, H, W]
        mask_hat = torch.stack(m_seq, dim=1)  # [B, T, 1, H, W]
        return z_hat, mask_hat
