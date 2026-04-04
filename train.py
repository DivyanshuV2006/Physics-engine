from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from datetime import datetime, timedelta
import time

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import KubricOcclusionDataset
from model import DepthRoutedLatentWorldModel

"""Training entrypoint for the depth-routed latent world model.

This script assembles data, model, optimizer, and losses into a standard
PyTorch loop so experiments are reproducible and easy to run from CLI.
"""


def _atomic_save_state_dict(model: nn.Module, output_path: str, temp_filename: str) -> None:
    """Atomically saves weights by staging to a temp file then replacing."""
    temp_path = os.path.join(os.path.dirname(output_path) or ".", temp_filename)
    torch.save(model.state_dict(), temp_path)
    os.replace(temp_path, output_path)


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
    if not (0.0 < args.val_split < 1.0):
        raise ValueError(f"--val-split must be in (0, 1), got {args.val_split}.")

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
        preload_depth_cache=True,
        preload_full_cache=True,
    )
    dataset_size = len(dataset)
    if dataset_size < 2:
        train_dataset = dataset
        val_dataset = None
        print("Dataset has <2 samples; validation split skipped.")
    else:
        val_size = int(round(dataset_size * args.val_split))
        val_size = max(1, min(val_size, dataset_size - 1))
        train_size = dataset_size - val_size
        split_gen = torch.Generator().manual_seed(args.split_seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_gen)
        print(f"Dataset split: train={train_size} ({100.0*train_size/dataset_size:.1f}%), "
              f"val={val_size} ({100.0*val_size/dataset_size:.1f}%)")
    effective_num_workers = args.num_workers
    if dataset.preload_full_cache and effective_num_workers > 0:
        print(
            "Full dataset cache + multi-worker loading enabled. "
            "This maximizes throughput on large-memory systems."
        )
    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": effective_num_workers,
        "pin_memory": True,
        "persistent_workers": effective_num_workers > 0,
    }
    if effective_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(**train_loader_kwargs)

    val_loader = None
    if val_dataset is not None:
        val_loader_kwargs = {
            "dataset": val_dataset,
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": effective_num_workers,
            "pin_memory": True,
            "persistent_workers": effective_num_workers > 0,
        }
        if effective_num_workers > 0:
            val_loader_kwargs["prefetch_factor"] = 2
        val_loader = DataLoader(**val_loader_kwargs)

    model = DepthRoutedLatentWorldModel(
        latent_channels=args.latent_channels,
        slot_dim=args.slot_dim,
        use_tqdm=args.model_tqdm and not args.disable_tqdm,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    tracked_modules = (
        ("physics_rnn", model.physics_rnn),
        ("visibility_head", model.visibility_head),
        ("position_head", model.position_head),
        ("sigma_head", model.sigma_head),
    )
    for module_name, module in tracked_modules:
        missing = [name for name, param in module.named_parameters() if id(param) not in opt_param_ids]
        if missing:
            raise RuntimeError(f"Optimizer is missing parameters for {module_name}: {missing}")
    print("Optimizer tracking verified for physics and mask branches.")

    # MSE supervises continuous latent reconstruction quality.
    latent_loss_fn = nn.MSELoss()
    # Weighted logits BCE penalizes missing sparse foreground pixels.
    pos_weight = torch.tensor([40.0], device=device)
    mask_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    trajectory_align_loss_fn = nn.SmoothL1Loss()

    output_path = "depth_routed_latent_world_model.pt"
    interrupt_output_path = "depth_routed_latent_world_model_interrupt.pt"
    model.train()
    train_start_time = time.time()
    training_completed = False
    interrupted = False
    unexpected_error: BaseException | None = None
    try:
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            epoch_latent_loss = 0.0
            epoch_mask_loss = 0.0
            epoch_total_loss = 0.0

            progress_ctx = (
                tqdm(train_loader, desc=f"Train {epoch + 1}/{args.epochs}", leave=False)
                if not args.disable_tqdm
                else nullcontext(train_loader)
            )
            with progress_ctx as progress:
                mask_collapse_detected = False
                for batch in progress:
                    z_0 = batch["z_0"].to(device)  # [B, C, H, W]
                    depth_map = batch["depth_map"].to(device)  # [B, 1, H, W]
                    trajectory = batch["trajectory"].to(device)  # [B, T, 2]
                    target_z = batch["target_z"].to(device)  # [B, T, C, H, W]
                    target_mask = batch["target_mask"].to(device)  # [B, T, 1, H, W]

                    optimizer.zero_grad(set_to_none=True)
                    z_hat, mask_logits, aux = model(
                        z_0=z_0, depth_map=depth_map, trajectory=trajectory, return_aux=True
                    )
                    pred_centers = aux["mask_centers"]  # [B, T, 2], same normalized coordinate system as trajectory

                    logits_min = float(mask_logits.detach().amin().item())
                    logits_max = float(mask_logits.detach().amax().item())
                    if logits_min < -10.0:
                        mask_collapse_detected = True
                    if (not args.disable_tqdm) and (progress.n == 0 or progress.n % 50 == 0):
                        print(
                            f"[logits] epoch={epoch + 1} step={progress.n} "
                            f"min={logits_min:.3f} max={logits_max:.3f}"
                        )

                    latent_loss = latent_loss_fn(z_hat, target_z)
                    mask_loss = mask_loss_fn(mask_logits, target_mask)
                    traj_align_loss = trajectory_align_loss_fn(pred_centers, trajectory)
                    # Soft range constraint (instead of hard tanh clamp): penalize only out-of-bounds coords.
                    traj_range_penalty = torch.relu(torch.abs(pred_centers) - 1.0).pow(2).mean()
                    total_loss = (
                        latent_loss
                        + mask_loss
                        + args.traj_align_weight * traj_align_loss
                        + args.traj_range_weight * traj_range_penalty
                    )

                    total_loss.backward()
                    if mask_collapse_detected:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_latent_loss += latent_loss.item()
                    epoch_mask_loss += mask_loss.item()
                    epoch_total_loss += total_loss.item()

                    if not args.disable_tqdm:
                        progress.set_postfix(
                            train_total=f"{total_loss.item():.4f}",
                        )

            train_num_batches = len(train_loader)
            train_latent_avg = epoch_latent_loss / train_num_batches
            train_mask_avg = epoch_mask_loss / train_num_batches
            train_total_avg = epoch_total_loss / train_num_batches

            val_latent_avg = float("nan")
            val_mask_avg = float("nan")
            val_total_avg = float("nan")
            if val_loader is not None:
                model.eval()
                val_latent_loss = 0.0
                val_mask_loss = 0.0
                val_total_loss = 0.0
                val_progress_ctx = (
                    tqdm(val_loader, desc=f"Val {epoch + 1}/{args.epochs}", leave=False)
                    if not args.disable_tqdm
                    else nullcontext(val_loader)
                )
                with torch.no_grad():
                    with val_progress_ctx as val_progress:
                        for batch in val_progress:
                            z_0 = batch["z_0"].to(device)
                            depth_map = batch["depth_map"].to(device)
                            trajectory = batch["trajectory"].to(device)
                            target_z = batch["target_z"].to(device)
                            target_mask = batch["target_mask"].to(device)

                            z_hat, mask_logits, aux = model(
                                z_0=z_0, depth_map=depth_map, trajectory=trajectory, return_aux=True
                            )
                            pred_centers = aux["mask_centers"]
                            latent_loss = latent_loss_fn(z_hat, target_z)
                            mask_loss = mask_loss_fn(mask_logits, target_mask)
                            traj_align_loss = trajectory_align_loss_fn(pred_centers, trajectory)
                            traj_range_penalty = torch.relu(torch.abs(pred_centers) - 1.0).pow(2).mean()
                            total_loss = (
                                latent_loss
                                + mask_loss
                                + args.traj_align_weight * traj_align_loss
                                + args.traj_range_weight * traj_range_penalty
                            )

                            val_latent_loss += latent_loss.item()
                            val_mask_loss += mask_loss.item()
                            val_total_loss += total_loss.item()

                            if not args.disable_tqdm:
                                val_progress.set_postfix(val_total=f"{total_loss.item():.4f}")

                val_num_batches = len(val_loader)
                val_latent_avg = val_latent_loss / val_num_batches
                val_mask_avg = val_mask_loss / val_num_batches
                val_total_avg = val_total_loss / val_num_batches
                model.train()

            epoch_elapsed = time.time() - epoch_start_time
            total_elapsed = time.time() - train_start_time
            avg_epoch_time = total_elapsed / (epoch + 1)
            remaining_epochs = args.epochs - (epoch + 1)
            eta_seconds = max(0.0, avg_epoch_time * remaining_epochs)
            eta_clock = datetime.now() + timedelta(seconds=eta_seconds)
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_latent={train_latent_avg:.6f} | "
                f"train_mask={train_mask_avg:.6f} | "
                f"train_total={train_total_avg:.6f} | "
                f"val_latent={val_latent_avg:.6f} | "
                f"val_mask={val_mask_avg:.6f} | "
                f"val_total={val_total_avg:.6f} | "
                f"epoch_time={epoch_elapsed:.1f}s | "
                f"eta={eta_seconds/60.0:.1f}m (finishes ~ {eta_clock.strftime('%H:%M:%S')})"
            )
        training_completed = True
    except KeyboardInterrupt:
        interrupted = True
    except BaseException as exc:
        unexpected_error = exc
    finally:
        try:
            # Always attempt to save a checkpoint via atomic swap.
            _atomic_save_state_dict(model, output_path, "temp_weights.pt")
            if interrupted:
                _atomic_save_state_dict(model, interrupt_output_path, "temp_interrupt_weights.pt")
                print(
                    f"\nTraining interrupted. Partial weights saved to {output_path} "
                    f"and {interrupt_output_path}"
                )
            elif unexpected_error is not None:
                _atomic_save_state_dict(model, interrupt_output_path, "temp_interrupt_weights.pt")
                print(
                    f"\nTraining stopped due to error; emergency weights saved to "
                    f"{output_path} and {interrupt_output_path}"
                )
            elif training_completed:
                print(f"\nTraining complete. Weights saved to {output_path}")
        except Exception as save_exc:
            print(f"\nWarning: failed to save checkpoint on exit: {save_exc}")

    if unexpected_error is not None:
        raise unexpected_error
    if interrupted:
        return


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
    parser.add_argument("--traj-align-weight", type=float, default=1.0, help="Weight for mask-center trajectory alignment loss.")
    parser.add_argument("--traj-range-weight", type=float, default=0.1, help="Weight for soft out-of-range coordinate penalty.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction (e.g. 0.2).")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for train/validation split.")
    parser.add_argument("--model-tqdm", action="store_true", help="Enable per-time-step model rollout tqdm")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable all tqdm progress bars")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
    