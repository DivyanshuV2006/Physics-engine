from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

"""Kubric synthetic data generator for occlusion-centric video sequences.

The generator creates deterministic scene layouts with semantic IDs so downstream
training can consume RGB, depth, and target-object visibility masks per frame.
"""

try:
    import kubric as kb
    from kubric.renderer import Blender
except ImportError as exc:
    raise ImportError(
        "Kubric is required for this script. Install it first (e.g. `pip install kubric`)."
    ) from exc


RESOLUTION = (256, 256)
FRAME_START = 1
FRAME_END = 24
FPS = 12

SEMANTIC_ID_FLOOR = 1
SEMANTIC_ID_WALL = 2
SEMANTIC_ID_TARGET = 3


def _build_scene(sequence_seed: int = 0) -> kb.Scene:
    """Creates one Kubric scene with floor, wall occluder, and moving sphere.

    What:
        Builds the exact scene graph required for object-permanence tests.
    How:
        Configures camera/light, adds semantic-labeled assets, and keyframes
        the target sphere from left to right behind the foreground wall.
    Why:
        Controlled geometry + semantics make occlusion behavior predictable and
        easy to supervise in the generated dataset.
    """
    rng = np.random.default_rng(sequence_seed)
    scene = kb.Scene(
        resolution=RESOLUTION,
        frame_start=FRAME_START,
        frame_end=FRAME_END,
        frame_rate=FPS,
    )

    # Perspective camera looking toward the object motion corridor.
    scene.camera = kb.PerspectiveCamera(
        name="camera",
        position=(0.0, -10.0, 4.0),
        look_at=(0.0, 0.0, 0.75),
    )
    scene += scene.camera

    # Directional key light for stable shading and silhouettes.
    sun = kb.DirectionalLight(
        name="sun",
        color=(1.0, 1.0, 1.0),
        intensity=3.5,
        position=(6.0, -8.0, 10.0),
        look_at=(0.0, 0.0, 0.0),
    )
    scene += sun

    floor = kb.Cube(
        name="floor",
        scale=(12.0, 12.0, 0.2),
        position=(0.0, 0.0, -0.2),
        static=True,
        material=kb.PrincipledBSDFMaterial(color=(0.75, 0.75, 0.75)),
    )
    floor.semantic_id = SEMANTIC_ID_FLOOR
    floor.segmentation_id = SEMANTIC_ID_FLOOR
    scene += floor

    wall = kb.Cube(
        name="wall_occluder",
        scale=(0.65, 1.8, 1.8),
        position=(0.0, -1.2, 1.0),
        static=True,
        material=kb.PrincipledBSDFMaterial(color=(0.90, 0.05, 0.05)),
    )
    wall.semantic_id = SEMANTIC_ID_WALL
    wall.segmentation_id = SEMANTIC_ID_WALL
    scene += wall

    # Slight per-sequence depth jitter keeps test sequences distinct.
    target_y = float(rng.uniform(0.6, 1.2))
    target = kb.Sphere(
        name="target",
        scale=0.45,
        position=(-4.0, target_y, 0.45),
        static=False,
        material=kb.PrincipledBSDFMaterial(color=(0.05, 0.25, 0.95)),
    )
    target.semantic_id = SEMANTIC_ID_TARGET
    target.segmentation_id = SEMANTIC_ID_TARGET
    scene += target

    # Animate target to pass behind the red wall from camera viewpoint.
    target.position = (-4.0, target_y, 0.45)
    target.keyframe_insert("position", FRAME_START)
    target.position = (4.0, target_y, 0.45)
    target.keyframe_insert("position", FRAME_END)

    return scene


def _select_segmentation_map(render_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Retrieves segmentation output from renderer results across API variants.

    Why:
        Different Kubric versions may expose either `semantic_segmentation` or
        `segmentation`; this compatibility layer keeps export logic stable.
    """
    # Kubric naming can vary with renderer/version. Prefer semantic if available.
    if "semantic_segmentation" in render_data:
        return render_data["semantic_segmentation"]
    if "segmentation" in render_data:
        return render_data["segmentation"]
    raise KeyError(
        "No segmentation output found in renderer data. "
        "Expected one of: semantic_segmentation, segmentation."
    )


def _save_sequence_frames(
    sequence_dir: Path,
    rgba_stack: np.ndarray,
    depth_stack: np.ndarray,
    segmentation_stack: np.ndarray,
) -> None:
    """Writes per-frame RGB, depth, and binary target masks to disk.

    How:
        - RGBA -> RGB PNG (`rgba_XXX.png`)
        - Depth -> float32 NPY (`depth_XXX.npy`) to preserve metric precision
        - Segmentation -> strict binary mask for semantic ID 3 (`mask_XXX.png`)
    Why:
        These exact modalities map directly to the training pipeline inputs and
        supervision targets.
    """
    sequence_dir.mkdir(parents=True, exist_ok=True)

    num_frames = rgba_stack.shape[0]
    for f in range(num_frames):
        rgba = rgba_stack[f]
        if rgba.shape[-1] == 4:
            rgb = rgba[..., :3]
        else:
            rgb = rgba
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        depth = np.asarray(depth_stack[f], dtype=np.float32)
        depth = np.squeeze(depth).astype(np.float32)

        seg = np.asarray(segmentation_stack[f])
        seg = np.squeeze(seg).astype(np.int32)
        mask = (seg == SEMANTIC_ID_TARGET).astype(np.uint8) * 255

        iio.imwrite(sequence_dir / f"rgba_{f:03d}.png", rgb)
        np.save(sequence_dir / f"depth_{f:03d}.npy", depth)
        iio.imwrite(sequence_dir / f"mask_{f:03d}.png", mask)


def generate_sequence(
    sequence_id: int,
    output_root: Path,
    scratch_root: Path | None = None,
) -> None:
    """Renders one full sequence and exports all required frame artifacts.

    Why validation checks exist:
        They fail fast when render outputs are incomplete or frame counts drift,
        preventing silent dataset corruption.
    """
    scene = _build_scene(sequence_seed=sequence_id)
    scratch_dir = scratch_root if scratch_root is not None else output_root / "_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    renderer = Blender(scene, scratch_dir=scratch_dir)
    render_data = renderer.render()

    if "rgba" not in render_data or "depth" not in render_data:
        missing = [k for k in ("rgba", "depth") if k not in render_data]
        raise KeyError(f"Missing renderer outputs: {missing}")

    rgba_stack = np.asarray(render_data["rgba"])
    depth_stack = np.asarray(render_data["depth"])
    segmentation_stack = np.asarray(_select_segmentation_map(render_data))

    expected_frames = FRAME_END - FRAME_START + 1
    for name, stack in (
        ("rgba", rgba_stack),
        ("depth", depth_stack),
        ("segmentation", segmentation_stack),
    ):
        if stack.shape[0] != expected_frames:
            raise ValueError(
                f"{name} has {stack.shape[0]} frames, expected {expected_frames}."
            )

    seq_dir = output_root / f"sequence_{sequence_id:04d}"
    _save_sequence_frames(seq_dir, rgba_stack, depth_stack, segmentation_stack)


def parse_args() -> argparse.Namespace:
    """Parses CLI options for dataset destination and sequence count."""
    parser = argparse.ArgumentParser(
        description="Generate Kubric synthetic sequences for 3D object permanence tests."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset"),
        help="Root output directory for generated sequences.",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=5,
        help="Number of sequences to generate.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=None,
        help="Optional scratch directory for Kubric/Blender temp files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Main loop intentionally keeps sequence generation explicit and serial so
    # failures can be traced to specific `sequence_id` values.
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for seq_id in tqdm(range(args.num_sequences), desc="Generating sequences"):
        generate_sequence(
            sequence_id=seq_id,
            output_root=args.output_root,
            scratch_root=args.scratch_root,
        )

    print(
        f"Generated {args.num_sequences} sequences in '{args.output_root.resolve()}'. "
        f"Each frame contains rgba_XXX.png, depth_XXX.npy, and mask_XXX.png."
    )
