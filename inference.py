import torch
import matplotlib.pyplot as plt
import argparse
from model import DepthRoutedLatentWorldModel
from dataset import KubricOcclusionDataset

def run_inference(args):
    print("Loading model and weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize the model with the exact same architecture as train.py
    model = DepthRoutedLatentWorldModel(
        latent_channels=args.latent_channels, 
        slot_dim=args.slot_dim
    ).to(device)
    
    # Load the saved weights
    model.load_state_dict(torch.load("depth_routed_latent_world_model.pt", map_location=device))
    model.eval() # Set to evaluation mode (turns off dropout/gradients)

    print("Loading test sequence...")
    # 2. Load a single sequence from the dataset to test
    dataset = KubricOcclusionDataset(
        root_dir=args.data_root, 
        sequence_length=args.sequence_length
    )
    
    # Grab the very first sequence in the dataset
    z_0, depth_map, trajectory, target_z, target_mask = dataset[0]

    # Add a batch dimension [B=1] and move to GPU
    z_0 = z_0.unsqueeze(0).to(device)
    depth_map = depth_map.unsqueeze(0).to(device)
    trajectory = trajectory.unsqueeze(0).to(device)

    print("Running forward pass...")
    # 3. Ask the model to predict the future masks
    with torch.no_grad():
        predicted_z, predicted_mask = model(z_0, depth_map, trajectory)

    # Convert tensors back to numpy arrays for plotting
    # Shape becomes [Time, Height, Width]
    pred_mask_np = predicted_mask.squeeze().cpu().numpy()
    target_mask_np = target_mask.squeeze().cpu().numpy()

    print("Generating visual proof...")
    # 4. Plot the Ground Truth vs. the Model's Prediction
    time_steps = args.sequence_length
    fig, axes = plt.subplots(2, time_steps, figsize=(2 * time_steps, 4))
    
    fig.suptitle("Object Permanence Proof: Ground Truth vs. Predicted Occlusion", fontsize=16, fontweight='bold', y=1.05)

    for t in range(time_steps):
        # Top Row: What actually happened (Ground Truth)
        axes[0, t].imshow(target_mask_np[t], cmap='gray', vmin=0, vmax=1)
        axes[0, t].set_title(f"True Mask (t={t+1})")
        axes[0, t].axis('off')

        # Bottom Row: What the neural network figured out (Prediction)
        axes[1, t].imshow(pred_mask_np[t], cmap='gray', vmin=0, vmax=1)
        axes[1, t].set_title(f"Predicted (t={t+1})")
        axes[1, t].axis('off')

    plt.tight_layout()
    plt.savefig("cvpr_visual_proof.png", dpi=300, bbox_inches='tight')
    print("Success! Saved 'cvpr_visual_proof.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visual Proof")
    parser.add_argument("--data-root", type=str, required=True, help="Path to your dataset folder")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--slot-dim", type=int, default=128)
    
    args = parser.parse_args()
    run_inference(args)