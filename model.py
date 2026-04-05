from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

"""Core model components for depth-routed latent prediction.

This file defines the full architecture that:
1) extracts an initial object-centric slot from the first latent frame,
2) evolves that slot through a trajectory-driven recurrent dynamics module,
3) gates visibility using depth comparison against the scene depth map, and
4) renders per-step latent predictions and masks.
"""


class SlotExtractor(nn.Module):
    """Extracts a compact slot state and an object depth estimate.

    What:
        Encodes `z_0` into a latent slot vector (`initial_slot`) and a scalar
        depth (`slot_depth`) that represents the object's relative depth.
    How:
        A lightweight CNN compresses spatial information, then an MLP maps the
        pooled features to `slot_dim + 1` outputs.
    Why:
        The slot is the persistent state used by the dynamics model, while the
        depth scalar is required by depth routing to infer visibility/occlusion.
    """

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
        """Runs slot extraction from the first latent observation."""
        x = self.encoder(z_0).flatten(1)  # [B, 128]
        out = self.mlp(x)  # [B, slot_dim + 1]
        initial_slot = out[:, :-1]  # [B, slot_dim]
        slot_depth = out[:, -1:]  # [B, 1]
        return initial_slot, slot_depth


class SpatialBroadcastDecoder(nn.Module):
    """Decodes a slot vector into a full latent feature map.

    What:
        Produces a spatial latent tensor `[B, C, H, W]` from slot states.
    How:
        Broadcasts the slot over the image plane, concatenates XY coordinates,
        and applies convolutional decoding.
    Why:
        Spatial broadcasting injects explicit positional context, which makes it
        easier for the decoder to reconstruct structured scene latents.
    """

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
        """Renders one latent frame from a slot state."""
        batch_size, slot_dim = slot.shape
        slot_map = slot.view(batch_size, slot_dim, 1, 1).expand(-1, -1, height, width)

        # Coordinate channels encode absolute pixel locations so the decoder can
        # learn spatially-aware structure from a globally broadcast slot.
        y_coords = torch.linspace(-1.0, 1.0, steps=height, device=slot.device, dtype=slot.dtype)
        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=slot.device, dtype=slot.dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coord_grid = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)

        decoded = self.decoder(torch.cat([slot_map, coord_grid], dim=1))  # [B, C, H, W]
        return decoded


class DepthRoutedLatentWorldModel(nn.Module):
    """
    What:
      End-to-end latent world model with depth-aware visibility routing.
    How:
      SlotExtractor -> PhysicsRNN (trajectory-conditioned rollout) ->
      DepthRouter (grid-sampled depth comparison) -> Spatial decoder.
    Why:
      Forces temporal predictions to respect occlusion constraints so objects
      remain persistent in latent dynamics even when hidden.

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
        self.velocity_multiplier = nn.Parameter(torch.tensor(1.0))
        self.renderer = SpatialBroadcastDecoder(slot_dim=slot_dim, out_channels=latent_channels)
        self.visibility_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim // 2, 1),
        )
        self.position_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim // 2, 2),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim // 2, 1),
        )
        # Explicit bias init for visibility branch to avoid deep-negative startup logits.
        nn.init.constant_(self.visibility_head[-1].bias, 0.0)
        # Start centered and with moderate spread in normalized coordinates.
        nn.init.constant_(self.position_head[-1].bias, 0.0)
        nn.init.constant_(self.sigma_head[-1].bias, 0.0)
        self.use_tqdm = use_tqdm

    @staticmethod
    def _route_depth(
        depth_map: torch.Tensor, trajectory_t: torch.Tensor, slot_depth: torch.Tensor
    ) -> torch.Tensor:
        """
        What:
            Computes per-sample visibility based on depth ordering.
        How:
            Uses `grid_sample` to read background depth at trajectory coordinates,
            then compares the sampled depth against predicted object depth.
        Why:
            A physically motivated visibility gate reduces impossible renderings
            where occluded objects leak through foreground geometry.

        depth_map: [B, 1, H, W]
        trajectory_t: [B, 2]
        slot_depth: [B, 1]
        Returns visibility logit per sample: [B, 1]
        """
        grid = trajectory_t.view(trajectory_t.shape[0], 1, 1, 2)  # [B, 1, 1, 2]
        bg_depth = F.grid_sample(
            depth_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(trajectory_t.shape[0], 1)  # [B, 1]

        # Positive logit => visible; negative logit => occluded.
        temperature = 8.0
        visible_logit = (bg_depth - slot_depth) * temperature  # [B, 1]
        return visible_logit

    @staticmethod
    def _spatial_stamp_logits(
        center_t: torch.Tensor,
        sigma_t: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Creates a per-step Gaussian spatial object prior in logit space.

        center_t: [B, 2] normalized coordinates in [-1, 1]
        sigma_t: [B, 1] positive spread parameter in normalized coordinates
        returns: [B, 1, H, W] logits
        """
        batch_size = center_t.shape[0]
        y_coords = torch.linspace(-1.0, 1.0, steps=height, device=center_t.device, dtype=center_t.dtype)
        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=center_t.device, dtype=center_t.dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        yy = yy.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]
        xx = xx.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]

        cx = center_t[:, 0].view(batch_size, 1, 1)  # [B,1,1]
        cy = center_t[:, 1].view(batch_size, 1, 1)  # [B,1,1]
        sigma = sigma_t.view(batch_size, 1, 1).clamp(min=0.05, max=0.5)  # [B,1,1]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2  # [B,H,W]

        spatial_prob = torch.exp(-dist2 / (2.0 * (sigma ** 2)))
        spatial_prob = spatial_prob.clamp(min=1e-4, max=1.0 - 1e-4)
        spatial_logits = torch.logit(spatial_prob).unsqueeze(1)  # [B,1,H,W]
        return spatial_logits

    def forward(
        self,
        z_0: torch.Tensor,
        depth_map: torch.Tensor,
        trajectory: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Rolls out depth-routed latent predictions across time.

        The recurrent state models object dynamics, while depth routing applies
        a visibility mask at each step before outputs are stacked as sequences.
        """
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
        center_seq = []

        time_iterator = range(time_steps)
        if self.use_tqdm:
            # Optional fine-grained progress when debugging long rollouts.
            time_iterator = tqdm(time_iterator, desc="Model rollout", leave=False)

        for t in time_iterator:
            traj_t = trajectory[:, t, :]  # [B, 2]
            # Zero-frame logic: first predicted frame uses initial slot directly.
            if t > 0:
                hidden = self.physics_rnn(self.velocity_multiplier * traj_t, hidden)  # [B, slot_dim]

            depth_visible_logit = self._route_depth(depth_map, traj_t, slot_depth)  # [B, 1]
            learned_visible_logit = self.visibility_head(hidden)  # [B, 1]
            visible_logit = depth_visible_logit + learned_visible_logit  # [B, 1]
            visible_alpha = torch.sigmoid(visible_logit).view(batch_size, 1, 1, 1)  # [B,1,1,1]

            # Physics RNN controls spatial placement and spread (fully differentiable).
            # Keep this branch linear (no hard tanh clamp) so gradients stay alive.
            center_delta = self.position_head(hidden)  # [B, 2]
            center_t = traj_t + 0.25 * center_delta  # [B, 2]
            # Sigma scaled for normalized coordinate space; small sphere footprint.
            sigma_t = 0.05 + 0.05 * torch.sigmoid(self.sigma_head(hidden))  # [B, 1] in [0.05, 0.10]

            spatial_logits = self._spatial_stamp_logits(
                center_t=center_t,
                sigma_t=sigma_t,
                height=height,
                width=width,
            )  # [B,1,H,W]
            spatial_alpha = torch.sigmoid(spatial_logits)  # [B,1,H,W]

            # Visibility scalar acts as intensity/alpha over a spatial object stamp.
            mask_prob = (visible_alpha * spatial_alpha).clamp(min=1e-4, max=1.0 - 1e-4)  # [B,1,H,W]
            mask_logits = torch.logit(mask_prob)  # [B,1,H,W]

            rendered_t = self.renderer(hidden, height, width)  # [B, C, H, W]
            routed_t = rendered_t * mask_prob  # [B, C, H, W]

            z_seq.append(routed_t)
            m_seq.append(mask_logits)
            center_seq.append(center_t)

        z_hat = torch.stack(z_seq, dim=1)  # [B, T, C, H, W]
        mask_hat = torch.stack(m_seq, dim=1)  # [B, T, 1, H, W]
        if not return_aux:
            return z_hat, mask_hat
        aux = {"mask_centers": torch.stack(center_seq, dim=1)}  # [B, T, 2]
        return z_hat, mask_hat, aux
