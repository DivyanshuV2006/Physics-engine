from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KubricOcclusionDataset
from model import DepthRoutedLatentWorldModel

"""Training entrypoint for the depth-routed latent world model.

This script assembles data, model, optimizer, and losses into a standard
PyTorch loop so experiments are reproducible and easy to run from CLI.
"""


def train(args: argparse.Namespace) -> None:
    """Runs supervised training on latent and occlusion-mask targets.

    What:
        Optimizes the model to predict future latent frames (`target_z`) and
        visibility masks (`target_mask`) simultaneously.
    How:
        Forward pass -> MSE(latents) + BCE(masks) -> backward -> AdamW step.
    Why:
        Joint optimization encourages both appearance consistency and explicit
        occlusion reasoning, which is central to object permanence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"Using GPU: {gpu_name}")
    else:
        print("Using CPU (CUDA not available).")

    dataset = KubricOcclusionDataset(
        root_dir=args.data_root,
        sequence_length=args.sequence_length,
        latent_channels=args.latent_channels,
        image_size=args.image_size,
        use_tqdm=not args.disable_tqdm,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DepthRoutedLatentWorldModel(
        latent_channels=args.latent_channels,
        slot_dim=args.slot_dim,
        use_tqdm=args.model_tqdm and not args.disable_tqdm,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # MSE supervises continuous latent reconstruction quality.
    latent_loss_fn = nn.MSELoss()
    # BCE supervises binary visibility probabilities for occlusion masks.
    mask_loss_fn = nn.BCELoss()

    model.train()
    for epoch in range(args.epochs):
        epoch_latent_loss = 0.0
        epoch_mask_loss = 0.0
        epoch_total_loss = 0.0

        progress_ctx = (
            tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
            if not args.disable_tqdm
            else nullcontext(loader)
        )
        with progress_ctx as progress:
            for batch in progress:
                z_0 = batch["z_0"].to(device)  # [B, C, H, W]
                depth_map = batch["depth_map"].to(device)  # [B, 1, H, W]
                trajectory = batch["trajectory"].to(device)  # [B, T, 2]
                target_z = batch["target_z"].to(device)  # [B, T, C, H, W]
                target_mask = batch["target_mask"].to(device)  # [B, T, 1, H, W]

                optimizer.zero_grad(set_to_none=True)
                z_hat, mask_hat = model(z_0=z_0, depth_map=depth_map, trajectory=trajectory)

                latent_loss = latent_loss_fn(z_hat, target_z)
                # Clamp avoids numerical issues in BCE with exact 0/1 probabilities.
                mask_hat_clamped = mask_hat.clamp(min=1e-4, max=1.0 - 1e-4)
                mask_loss = mask_loss_fn(mask_hat_clamped, target_mask)
                total_loss = latent_loss + mask_loss

                total_loss.backward()
                optimizer.step()

                epoch_latent_loss += latent_loss.item()
                epoch_mask_loss += mask_loss.item()
                epoch_total_loss += total_loss.item()

                if not args.disable_tqdm:
                    progress.set_postfix(
                        latent_loss=f"{latent_loss.item():.4f}",
                        mask_loss=f"{mask_loss.item():.4f}",
                        total_loss=f"{total_loss.item():.4f}",
                    )

        num_batches = len(loader)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"latent_loss={epoch_latent_loss / num_batches:.6f} | "
            f"mask_loss={epoch_mask_loss / num_batches:.6f} | "
            f"total_loss={epoch_total_loss / num_batches:.6f}"
        )
    # Save the final model weights so the professor can send them back to you
    output_path = "depth_routed_latent_world_model.pt"
    torch.save(model.state_dict(), output_path)
    print(f"\nTraining complete. Weights saved to {output_path}")


def parse_args() -> argparse.Namespace:
    """Defines CLI arguments for reproducible experiment configuration."""
    parser = argparse.ArgumentParser(description="Train Depth-Routed Latent World Model")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory of Kubric-style scenes")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--slot-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-tqdm", action="store_true", help="Enable per-time-step model rollout tqdm")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable all tqdm progress bars")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
    