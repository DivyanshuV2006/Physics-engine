# Physics-engine
## Project Description

This project is a synthetic dataset generator using the [Kubric](https://github.com/google-research/kubric) physics engine and PyTorch. It creates and processes scenes with occluded objects, producing data useful for computer vision tasks such as occlusion reasoning and depth estimation.

**Features:**
- Generates scenes with RGB(A) images, depth maps, and binary masks.
- Provides a PyTorch `Dataset` class for easy integration with machine learning workflows.
- Includes a simple convolutional VAE encoder for image-to-latent representations.
- Designed for GPU-accelerated environments via Docker and NVIDIA runtime.

**How to use:**
1. Place your synthetic scene directories in the required structure.
2. Use the provided Dockerfile to build the reproducible environment.
3. Run the generator using Docker Compose.

**Requirements:**
- Python (with numpy, torch, PIL, tqdm)
- Docker with NVIDIA GPU support
- Kubric SDK

See `dataset.py` for the dataset implementation and usage details.

## File-by-file design notes (what/how/why)

- `dataset-generator.py`
  - **What:** Generates Kubric sequences with RGB, depth, and binary target masks.
  - **How:** Builds a fixed scene graph, animates a target object, renders all frames, and exports frame-wise artifacts.
  - **Why:** Produces controlled occlusion-heavy data for object permanence experiments.

- `dataset.py`
  - **What:** Loads synthetic sequences and returns model-ready tensors.
  - **How:** Discovers valid folders, encodes RGB into latents, resizes depth/mask maps, and emits a mock trajectory.
  - **Why:** Keeps training input preparation consistent and centralized.

- `model.py`
  - **What:** Implements the depth-routed latent world model architecture.
  - **How:** Extracts a slot from `z_0`, rolls it with GRU over trajectory, applies depth-based visibility routing, and decodes latents.
  - **Why:** Enforces physically plausible occlusion behavior in temporal latent prediction.

- `train.py`
  - **What:** Runs end-to-end training.
  - **How:** Creates dataloader/model/optimizer, computes latent MSE + mask BCE losses, and updates weights each step.
  - **Why:** Provides a reproducible baseline training entrypoint.