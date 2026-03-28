from __future__ import annotations

import os
from glob import glob
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

"""Dataset utilities for synthetic Kubric occlusion sequences.

This file handles discovery, loading, and lightweight preprocessing so the
training loop receives tensors with consistent shapes for latent prediction and
occlusion supervision.
"""


def resolve_device(verbose: bool = True) -> torch.device:
    """Selects CUDA by default and falls back to CPU when unavailable."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Device] Using CUDA: {gpu_name}")
        return device
    device = torch.device("cpu")
    if verbose:
        print("[Device] CUDA not found. Using CPU.")
    return device


class DummyVAEEncoder(nn.Module):
    """Mock image encoder used as a stand-in for a real VAE encoder.

    What:
        Converts RGB images into latent feature maps.
    How:
        A small stride-based CNN downsamples the image and projects features
        into `latent_channels`.
    Why:
        Keeps the pipeline executable before integrating a production VAE.
    """

    def __init__(self, in_channels: int = 3, latent_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies CNN downsampling and feature projection."""
        return self.net(x)


class KubricOcclusionDataset(Dataset):
    """
    What:
      Loads Kubric-generated image/depth/mask sequences and returns model-ready
      tensors for latent prediction and mask supervision.
    How:
      Discovers valid scene folders, loads frame assets, encodes images into
      latents, resizes depth/mask to latent resolution, and builds a mock
      trajectory.
    Why:
      Centralizes I/O + preprocessing so model/training code remains clean and
      shape-consistent.

    Expected synthetic scene directory contents:
      - rgba_*.png
      - depth_*.npy
      - mask_*.png

    Returned dictionary keys:
      - z_0: [C, H, W]
      - depth_map: [1, H, W]
      - trajectory: [T, 2] in [-1, 1]
      - target_z: [T, C, H, W]
      - target_mask: [T, 1, H, W] binary {0.0, 1.0}
    """

    def __init__(
        self,
        root_dir: str,
        sequence_length: int = 8,
        latent_channels: int = 32,
        image_size: int = 128,
        use_tqdm: bool = False,
        device: torch.device | str | None = None,
        verbose_device: bool = False,
    ) -> None:
        """Initializes dataset paths, settings, and mock encoder."""
        super().__init__()
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.use_tqdm = use_tqdm
        self.device = torch.device(device) if device is not None else resolve_device(verbose=verbose_device)

        self.encoder = DummyVAEEncoder(in_channels=3, latent_channels=latent_channels)
        self.encoder.to(self.device)
        self.encoder.eval()

        self.sequences = self._discover_sequences(root_dir)
        if not self.sequences:
            raise RuntimeError(
                f"No valid sequences found under '{root_dir}'. "
                "Each scene folder must contain rgba_*.png, depth_*.npy, and mask_*.png."
            )

    @staticmethod
    def _sorted_files(scene_dir: str, pattern: str) -> List[str]:
        """Returns deterministic file ordering for temporal consistency."""
        return sorted(glob(os.path.join(scene_dir, pattern)))

    def _discover_sequences(self, root_dir: str) -> List[Dict[str, List[str]]]:
        """Scans for directories that contain all required modality files.

        Why this matters:
            Training assumes aligned rgba/depth/mask streams, so we only keep
            folders that have all three modalities.
        """
        sequences: List[Dict[str, List[str]]] = []
        walk_iter = os.walk(root_dir)
        if self.use_tqdm:
            walk_iter = tqdm(walk_iter, desc="Scanning dataset", leave=False)

        for current_dir, _, _ in walk_iter:
            rgba_files = self._sorted_files(current_dir, "rgba_*.png")
            depth_files = self._sorted_files(current_dir, "depth_*.npy")
            mask_files = self._sorted_files(current_dir, "mask_*.png")

            if rgba_files and depth_files and mask_files:
                sequences.append(
                    {
                        "rgba": rgba_files,
                        "depth": depth_files,
                        "mask": mask_files,
                    }
                )
        return sequences

    def __len__(self) -> int:
        """Number of discovered sequence folders."""
        return len(self.sequences)

    def _load_rgba_as_rgb_tensor(self, path: str) -> torch.Tensor:
        """Loads an RGBA PNG, drops alpha, normalizes to [0, 1], returns CHW."""
        img = Image.open(path).convert("RGBA").resize((self.image_size, self.image_size), Image.BILINEAR)
        rgba = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 4]
        rgb = rgba[..., :3]
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # [3, H, W]
        return tensor

    @staticmethod
    def _load_depth(path: str) -> torch.Tensor:
        """Loads exact float depth from .npy without quantization losses."""
        depth = np.load(path).astype(np.float32)  # exact floats from npy
        if depth.ndim == 3:
            depth = depth[..., 0]
        return torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]

    @staticmethod
    def _load_mask(path: str, out_h: int, out_w: int) -> torch.Tensor:
        """Loads a binary mask and resizes with nearest-neighbor to preserve labels."""
        mask = Image.open(path).convert("L")
        mask_np = np.asarray(mask, dtype=np.uint8)
        mask_bin = (mask_np > 127).astype(np.float32)  # [H, W], binary
        mask_t = torch.from_numpy(mask_bin).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        mask_t = F.interpolate(mask_t, size=(out_h, out_w), mode="nearest")
        return mask_t.squeeze(0)  # [1, H, W]

    def _encode_image(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Encodes one RGB frame into latent space using the dummy encoder."""
        with torch.no_grad():
            z = self.encoder(rgb_tensor.unsqueeze(0).to(self.device))  # [1, C, H, W]
        return z.squeeze(0).cpu()  # [C, H, W]

    def _mock_trajectory(self) -> torch.Tensor:
        """Creates a simple horizontal straight-line trajectory in normalized coords."""
        x = torch.linspace(-0.8, 0.8, steps=self.sequence_length)
        y = torch.zeros_like(x)
        return torch.stack([x, y], dim=-1)  # [T, 2]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Builds one training sample with aligned inputs and future targets.

        Why frame indexing works this way:
            We use frame 0 as conditioning (`z_0`) and frames 1..T as targets.
            If a sequence is short, the final frame is repeated to keep shapes
            fixed for batching.
        """
        seq = self.sequences[idx]
        rgba_files = seq["rgba"]
        depth_files = seq["depth"]
        mask_files = seq["mask"]

        # z_0 from first frame.
        z_0 = self._encode_image(self._load_rgba_as_rgb_tensor(rgba_files[0]))  # [C, H, W]
        latent_h, latent_w = z_0.shape[-2], z_0.shape[-1]

        # depth_map from first depth frame, resized to latent resolution.
        depth_map = self._load_depth(depth_files[0]).unsqueeze(0)  # [1, 1, H, W]
        depth_map = F.interpolate(depth_map, size=(latent_h, latent_w), mode="bilinear", align_corners=True)
        depth_map = depth_map.squeeze(0)  # [1, H, W]

        # Future targets use frames 1..T when available, otherwise repeat last.
        target_z_list: List[torch.Tensor] = []
        target_mask_list: List[torch.Tensor] = []
        for t in range(1, self.sequence_length + 1):
            rgba_idx = min(t, len(rgba_files) - 1)
            mask_idx = min(t, len(mask_files) - 1)

            z_t = self._encode_image(self._load_rgba_as_rgb_tensor(rgba_files[rgba_idx]))  # [C, H, W]
            m_t = self._load_mask(mask_files[mask_idx], out_h=latent_h, out_w=latent_w)  # [1, H, W]

            target_z_list.append(z_t)
            target_mask_list.append(m_t)

        target_z = torch.stack(target_z_list, dim=0)  # [T, C, H, W]
        target_mask = torch.stack(target_mask_list, dim=0)  # [T, 1, H, W]
        trajectory = self._mock_trajectory()  # [T, 2]

        return {
            "z_0": z_0.float(),
            "depth_map": depth_map.float(),
            "trajectory": trajectory.float(),
            "target_z": target_z.float(),
            "target_mask": target_mask.float(),
        }
