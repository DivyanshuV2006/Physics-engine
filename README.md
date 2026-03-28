# Depth-Routed Latent World Model

This repository contains the official PyTorch implementation and synthetic dataset generation pipeline for the **Depth-Routed Latent World Model**. 

This project explores a novel architecture for 3D object permanence in neural networks. By integrating explicit depth-map routing and differentiable visibility masking into a latent dynamics model, the network learns to track moving objects even when they are 100% occluded by foreground geometry.

## 🧠 Core Architecture

Traditional latent world models struggle with object permanence because they rely purely on spatial reconstruction. When an object is occluded, its pixels disappear, often causing the model's recurrent state to "forget" the object entirely. 

This architecture solves this by enforcing geometric visibility rules on the latent features:

1. **Slot Extractor:** A lightweight CNN encoder compresses the initial visual observation (`z_0`) into a persistent latent slot and predicts a scalar relative depth for the object.
2. **Physics RNN:** A trajectory-conditioned GRU cell evolves the latent slot across time, tracking the object's physical dynamics independently of its visibility.
3. **Differentiable Depth Router:** The core innovation. The model samples the static background depth map at the object's current trajectory coordinates and compares it to the object's predicted depth. A steep sigmoid function acts as a differentiable visibility gate, dropping the output to 0.0 when the object is geometrically behind the foreground.
4. **Spatial Broadcast Decoder:** The routed latent state is explicitly broadcast across an XY coordinate grid to reconstruct the predicted spatial latent maps and binary occlusion masks.

## 🗄️ The Kubric Occlusion Dataset

To mathematically isolate and prove the depth-routing logic, this repository includes a custom, containerized physics pipeline built on Google's **Kubric** engine. 

The current dataset generator outputs a controlled scenario: a dynamic target sphere passing horizontally behind a static foreground occluder (wall). 

The pipeline strictly exports three modalities per sequence to enforce multi-objective supervision:
* `rgba_*.png`: The visual input.
* `depth_*.npy`: Uncompressed float32 arrays preserving exact sub-pixel depth measurements.
* `mask_*.png`: Binary visibility targets for the BCE occlusion loss.

## 🚀 Quick Start

### 1. Data Generation (Docker)
The dataset generation relies on Blender's rendering engine and is containerized to prevent dependency conflicts. 

Generate the data locally using the provided Docker environment:
```bash
docker run --rm -v "${PWD}:/workspace" kubricdocker/kubric python /workspace/dataset-generator.py --output-root dataset --num-sequences 2000
```

### 2. Training (Local or Cloud GPU)
The training loop is purely PyTorch and does not require Kubric. It includes automatic epoch checkpointing to survive preemptible cloud instances (like Google Colab).

Install dependencies and execute the training loop:
```bash
pip install -r requirements.txt
python train.py --data-root dataset --epochs 20 --batch-size 32
```

## 🗺️ Roadmap & Future Work

This codebase currently serves as a foundational proof-of-concept for the core depth-routing mathematics. During the upcoming summer break, this architecture will be scaled significantly:
* **Complex Mesh Integration:** Swapping the primitive spherical targets for high-fidelity `.obj` and `.gltf` assets.
* **Dynamic Occluders:** Training the network to route depth against moving foreground elements rather than a static environment.
* **Compute Scaling:** Transitioning the data generation and training loops from a local environment to a dedicated multi-GPU compute cluster.
