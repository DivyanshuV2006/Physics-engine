import matplotlib.pyplot as plt
import re
import argparse

def plot_training_curves(log_file):
    epochs = []
    latent_losses = []
    mask_losses = []

    # Parse the text file line by line
    with open(log_file, 'r') as f:
        for line in f:
            # Use regex to find lines containing our specific PyTorch log format
            match = re.search(r"Epoch (\d+)/.*latent_loss=([0-9.]+)\s*\|\s*mask_loss=([0-9.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                latent_losses.append(float(match.group(2)))
                mask_losses.append(float(match.group(3)))

    if not epochs:
        print("Error: Could not find any training logs in the provided file.")
        return

    # Set up a professional, side-by-side graph layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. The Physics/Latent Tracking Curve
    ax1.plot(epochs, latent_losses, color='#1f77b4', linewidth=2, label='Latent Tracking Loss')
    ax1.set_title('Physics RNN Convergence', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 2. The Occlusion/Depth Routing Curve
    ax2.plot(epochs, mask_losses, color='#d62728', linewidth=2, label='Occlusion Mask Loss')
    ax2.set_title('Depth Router Convergence', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Binary Cross Entropy (BCE)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # Save the final image
    plt.tight_layout()
    output_filename = 'training_convergence.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Graph saved as {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PyTorch Training Logs")
    parser.add_argument("log_file", type=str, help="Path to your text file containing the Colab terminal output")
    args = parser.parse_args()
    
    plot_training_curves(args.log_file)