import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np
from model import DepthRoutedLatentWorldModel
from dataset import KubricOcclusionDataset


def _select_sequence_index(dataset: KubricOcclusionDataset, sample_index: int, scan_limit: int) -> int:
    """Returns a dataset index with visible target masks for a useful plot."""
    if sample_index >= 0:
        if sample_index >= len(dataset):
            raise IndexError(f"--sample-index={sample_index} is out of range for dataset size {len(dataset)}")
        return sample_index

    limit = min(len(dataset), max(1, scan_limit))
    best_idx = 0
    best_score = -1.0
    for idx in range(limit):
        sample = dataset[idx]
        score = float(sample["target_mask"].sum().item())
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def run_inference(args):
    print("Loading model and weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("Using CPU (CUDA not available).")
    
    # 1. Initialize the model with the exact same architecture as train.py
    model = DepthRoutedLatentWorldModel(
        latent_channels=args.latent_channels, 
        slot_dim=args.slot_dim
    ).to(device)
    
    # Load the saved weights
    weights_path = Path(args.weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # Set to evaluation mode (turns off dropout/gradients)

    print("Loading test sequence...")
    # 2. Load a single sequence from the dataset to test
    dataset = KubricOcclusionDataset(
        root_dir=args.data_root, 
        sequence_length=args.sequence_length
    )
    
    # Grab one sequence from the dataset.
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; no sequences available for inference.")
    selected_idx = _select_sequence_index(dataset, args.sample_index, args.scan_limit)
    print(f"Using sequence index {selected_idx}")
    sample = dataset[selected_idx]
    z_0 = sample["z_0"]
    depth_map = sample["depth_map"]
    trajectory = sample["trajectory"]
    target_mask = sample["target_mask"]

    # Add a batch dimension [B=1] and move to GPU
    z_0 = z_0.unsqueeze(0).to(device)
    depth_map = depth_map.unsqueeze(0).to(device)
    trajectory = trajectory.unsqueeze(0).to(device)

    print("Running forward pass...")
    # 3. Ask the model to predict the future masks
    with torch.no_grad():
        predicted_z, predicted_mask_logits = model(z_0, depth_map, trajectory)
        predicted_mask = torch.sigmoid(predicted_mask_logits)

    # Convert tensors back to numpy arrays for plotting
    # Shape becomes [Time, Height, Width]
    pred_mask_np = predicted_mask.squeeze(0).squeeze(1).cpu().numpy()  # [T, H, W]
    target_mask_np = target_mask.squeeze(1).cpu().numpy()  # [T, H, W]
    pred_mask_bin = (pred_mask_np >= args.pred_threshold).astype(np.float32)
    pred_std = float(pred_mask_np.std())
    print(
        f"Mask stats | target[min={target_mask_np.min():.4f}, max={target_mask_np.max():.4f}] "
        f"pred[min={pred_mask_np.min():.4f}, max={pred_mask_np.max():.4f}]"
    )
    print(
        f"Foreground ratio | target={float(target_mask_np.mean()):.4f} "
        f"pred_bin={float(pred_mask_bin.mean()):.4f} (threshold={args.pred_threshold:.2f})"
    )
    if pred_std < 1e-4:
        print(
            "Warning: predicted masks are nearly constant (collapsed). "
            "Weights are being used, but the model output has very low variance."
        )

    print("Generating visual proof...")
    # 4. Plot the Ground Truth vs. the Model's Prediction
    time_steps = min(args.sequence_length, pred_mask_np.shape[0], target_mask_np.shape[0])
    print("Temporal alignment: plotting pred[t] directly against target[t] (first output -> t=1 slot).")
    fig, axes = plt.subplots(2, time_steps, figsize=(2.8 * time_steps, 6.0), squeeze=False)
    
    fig.suptitle("Object Permanence Proof: Ground Truth vs. Predicted Occlusion", fontsize=16, fontweight='bold', y=1.05)
    axes[0, 0].set_ylabel("Ground Truth", fontsize=11)
    axes[1, 0].set_ylabel("Prediction", fontsize=11)

    for t in range(time_steps):
        # Top Row: What actually happened (Ground Truth)
        axes[0, t].imshow(target_mask_np[t], cmap='gray', vmin=0, vmax=1)
        axes[0, t].set_title(f"True Mask (t={t+1})")
        axes[0, t].set_xticks([])
        axes[0, t].set_yticks([])
        axes[0, t].set_facecolor("#1e1e1e")

        # Bottom Row: What the neural network figured out (Prediction)
        pred_vmin = float(pred_mask_np.min())
        pred_vmax = float(pred_mask_np.max())
        if abs(pred_vmax - pred_vmin) < 1e-8:
            pred_vmin, pred_vmax = 0.0, 1.0
        im = axes[1, t].imshow(pred_mask_np[t], cmap='magma', vmin=pred_vmin, vmax=pred_vmax)
        axes[1, t].set_title(f"Pred Prob (t={t+1})")
        axes[1, t].set_xticks([])
        axes[1, t].set_yticks([])
        axes[1, t].set_facecolor("#1e1e1e")

        # Keep highly visible frames so both rows are always unmistakable.
        for spine in axes[0, t].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_edgecolor("#2f80ed")  # blue frame for ground truth row
        for spine in axes[1, t].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_edgecolor("#f2994a")  # orange frame for prediction row

    plt.subplots_adjust(wspace=0.30, hspace=0.55, top=0.86, bottom=0.08)
    cbar = fig.colorbar(im, ax=axes[1, :], fraction=0.02, pad=0.02)
    cbar.set_label("Predicted mask probability")
    output_path = Path(args.output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Success! Saved '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visual Proof")
    parser.add_argument("--data-root", type=str, required=True, help="Path to your dataset folder")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--slot-dim", type=int, default=128)
    parser.add_argument("--weights-path", type=str, default="depth_routed_latent_world_model.pt")
    parser.add_argument("--output-path", type=str, default="cvpr_visual_proof.png")
    parser.add_argument("--sample-index", type=int, default=-1, help="Dataset index to visualize; -1 auto-selects a visible sequence.")
    parser.add_argument("--scan-limit", type=int, default=32, help="How many sequences to scan when auto-selecting.")
    parser.add_argument("--pred-threshold", type=float, default=0.5, help="Threshold used to binarize predicted masks for visualization.")
    
    args = parser.parse_args()
    run_inference(args)