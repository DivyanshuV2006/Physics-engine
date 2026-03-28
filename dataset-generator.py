from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm

"""Kubric synthetic data generator for occlusion-centric video sequences.

Primary goal:
    Generate temporally correct RGB, depth, and target masks where a blue sphere
    moves from x=-4 to x=4 behind a red foreground occluder.
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
NUM_FRAMES = FRAME_END - FRAME_START + 1
RENDER_LAYERS = ("rgba", "depth", "segmentation")

SEMANTIC_ID_FLOOR = 1
SEMANTIC_ID_WALL = 2
SEMANTIC_ID_TARGET = 3

TARGET_X_START = -4.0
TARGET_X_END = 4.0
TARGET_Z = 0.45


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


def _target_x_for_frame(frame: int) -> float:
    """Returns the expected target x-position for a given frame index."""
    alpha = (frame - FRAME_START) / max(1, (FRAME_END - FRAME_START))
    return float(TARGET_X_START + alpha * (TARGET_X_END - TARGET_X_START))


def _set_manual_target_animation(target: kb.Object3D, target_y: float) -> None:
    """Animates the target sphere deterministically across all frames.

    Why this function exists:
        Some scene configurations only keyframe the endpoints. Depending on
        renderer/asset state, this can appear frozen. Explicitly writing a
        keyframe for every frame guarantees movement without physics simulation.

    Implementation detail:
        The target is treated as a kinematic/static object (`static=True`) and
        follows manually authored position keyframes only.
    """
    x_positions = np.linspace(TARGET_X_START, TARGET_X_END, num=NUM_FRAMES, dtype=np.float32)
    for idx, frame in enumerate(range(FRAME_START, FRAME_END + 1)):
        target.position = (float(x_positions[idx]), target_y, TARGET_Z)
        target.keyframe_insert("position", frame=frame)


def _build_scene(
    sequence_seed: int = 0,
    target_x: float | None = None,
    use_keyframes: bool = True,
) -> kb.Scene:
    """Creates one scene with floor, wall occluder, and manually animated target."""
    rng = np.random.default_rng(sequence_seed)
    scene = kb.Scene(
        resolution=RESOLUTION,
        frame_start=FRAME_START,
        frame_end=FRAME_END,
        frame_rate=FPS,
    )

    # Camera sees the motion corridor and occluder clearly.
    scene.camera = kb.PerspectiveCamera(
        name="camera",
        position=(0.0, -10.0, 4.0),
        look_at=(0.0, 0.0, 0.9),
    )
    scene += scene.camera

    # Directional key light plus a soft fill keeps object silhouettes stable.
    sun = kb.DirectionalLight(
        name="sun",
        color=(1.0, 1.0, 1.0),
        intensity=3.5,
        position=(6.0, -8.0, 10.0),
        look_at=(0.0, 0.0, 0.0),
    )
    fill = kb.DirectionalLight(
        name="fill",
        color=(1.0, 1.0, 1.0),
        intensity=1.0,
        position=(-6.0, -6.0, 6.0),
        look_at=(0.0, 0.0, 0.0),
    )
    scene += sun
    scene += fill

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
        scale=(0.7, 1.7, 1.8),
        position=(0.0, -0.9, 1.0),
        static=True,
        material=kb.PrincipledBSDFMaterial(color=(0.90, 0.05, 0.05)),
    )
    wall.semantic_id = SEMANTIC_ID_WALL
    wall.segmentation_id = SEMANTIC_ID_WALL
    scene += wall

    # Keep y near center so the trajectory always passes behind the wall.
    target_y = float(rng.uniform(0.95, 1.25))
    target = kb.Sphere(
        name="target",
        scale=0.45,
        position=(TARGET_X_START if target_x is None else target_x, target_y, TARGET_Z),
        # Keep non-static so Blender evaluates transform keyframes correctly.
        # We still bypass physics simulation entirely; motion is keyframe-driven.
        static=False,
        material=kb.PrincipledBSDFMaterial(color=(0.05, 0.25, 0.95)),
    )
    target.semantic_id = SEMANTIC_ID_TARGET
    target.segmentation_id = SEMANTIC_ID_TARGET
    scene += target

    if use_keyframes:
        _set_manual_target_animation(target, target_y)
    return scene


def _select_segmentation_map(render_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Retrieves segmentation map from renderer output across Kubric versions."""
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
    mask_stack: np.ndarray,
) -> None:
    """Writes final deliverables only (png/npy/png) into sequence directory.

    Important:
        This function does not write EXR files. Any renderer scratch EXR output
        stays in a separate scratch directory and never enters `sequence_XXXX`.
    """
    sequence_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(rgba_stack.shape[0]):
        rgba = np.asarray(rgba_stack[frame_idx])
        rgb = rgba[..., :3] if rgba.shape[-1] == 4 else rgba
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        depth = np.asarray(depth_stack[frame_idx], dtype=np.float32)
        depth = np.squeeze(depth).astype(np.float32)

        mask = np.asarray(mask_stack[frame_idx], dtype=np.uint8)
        mask = np.squeeze(mask)

        iio.imwrite(sequence_dir / f"rgba_{frame_idx:03d}.png", rgb)
        np.save(sequence_dir / f"depth_{frame_idx:03d}.npy", depth)
        iio.imwrite(sequence_dir / f"mask_{frame_idx:03d}.png", mask)


def _infer_target_instance_id(segmentation_stack: np.ndarray) -> int:
    """Infers the target instance ID from instance segmentation.

    Blender renderer output is instance-indexed rather than semantic-indexed in
    this environment. The target sphere is the smallest persistent foreground
    object, so we pick the non-zero ID with the smallest max pixel footprint.
    """
    seg = np.asarray(segmentation_stack).squeeze(-1).astype(np.int32)  # [T,H,W]
    unique_ids = [int(i) for i in np.unique(seg) if int(i) != 0]
    if not unique_ids:
        raise ValueError("No non-background instance IDs found in segmentation.")

    best_id = unique_ids[0]
    best_max_area = None
    for instance_id in unique_ids:
        per_frame_areas = np.sum(seg == instance_id, axis=(1, 2))
        max_area = int(np.max(per_frame_areas))
        if best_max_area is None or max_area < best_max_area:
            best_max_area = max_area
            best_id = instance_id
    return best_id


def _extract_stacks(render_data: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts RGBA/depth and derives a binary target mask from segmentation."""
    if "rgba" not in render_data or "depth" not in render_data:
        missing = [k for k in ("rgba", "depth") if k not in render_data]
        raise KeyError(f"Missing renderer outputs: {missing}")

    rgba_stack = np.asarray(render_data["rgba"])
    depth_stack = np.asarray(render_data["depth"])
    segmentation_stack = np.asarray(_select_segmentation_map(render_data))

    target_instance_id = _infer_target_instance_id(segmentation_stack)
    seg = np.asarray(segmentation_stack).squeeze(-1).astype(np.int32)
    mask_stack = (seg == target_instance_id).astype(np.uint8) * 255  # [T,H,W]
    return rgba_stack, depth_stack, mask_stack


def _render_full_sequence(scene: kb.Scene, scratch_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Renders a keyframed sequence in one pass using only required layers."""
    renderer = Blender(scene, scratch_dir=scratch_dir)
    render_data = renderer.render(
        frames=list(range(FRAME_START, FRAME_END + 1)),
        return_layers=RENDER_LAYERS,
    )
    return _extract_stacks(render_data)


def _has_temporal_motion(rgba_stack: np.ndarray, eps: float = 1e-3) -> bool:
    """Returns True when first and last rendered frames differ."""
    if rgba_stack.shape[0] < 2:
        return False
    a = rgba_stack[0].astype(np.float32)
    b = rgba_stack[-1].astype(np.float32)
    mean_abs_diff = float(np.mean(np.abs(a - b)))
    return mean_abs_diff > eps


def _render_fallback_framewise(sequence_seed: int, scratch_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback renderer that guarantees motion by rendering frame-by-frame.

    Used only when keyframed full-sequence rendering appears frozen.
    """
    rgba_frames = []
    depth_frames = []
    mask_frames = []

    for frame in range(FRAME_START, FRAME_END + 1):
        x = _target_x_for_frame(frame)
        scene = _build_scene(sequence_seed=sequence_seed, target_x=x, use_keyframes=False)
        scene.frame_start = FRAME_START
        scene.frame_end = FRAME_START
        scene.frame_rate = FPS
        frame_scratch = scratch_dir / f"fallback_{frame:03d}"
        frame_scratch.mkdir(parents=True, exist_ok=True)

        renderer = Blender(scene, scratch_dir=frame_scratch)
        render_data = renderer.render(frames=[FRAME_START], return_layers=RENDER_LAYERS)
        rgba_stack, depth_stack, mask_stack = _extract_stacks(render_data)

        rgba_frames.append(rgba_stack[0])
        depth_frames.append(depth_stack[0])
        mask_frames.append(mask_stack[0])

    return np.stack(rgba_frames, axis=0), np.stack(depth_frames, axis=0), np.stack(mask_frames, axis=0)


def generate_sequence(
    sequence_id: int,
    output_root: Path,
    scratch_root: Path | None = None,
    render_mode: str = "fallback",
) -> None:
    """Renders one sequence and exports RGB/depth/binary-mask frame triplets."""
    scene = _build_scene(sequence_seed=sequence_id, use_keyframes=True)
    scratch_dir = scratch_root if scratch_root is not None else output_root / "_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    if render_mode == "full":
        rgba_stack, depth_stack, mask_stack = _render_full_sequence(scene, scratch_dir=scratch_dir)
    elif render_mode == "fallback":
        rgba_stack, depth_stack, mask_stack = _render_fallback_framewise(
            sequence_seed=sequence_id, scratch_dir=scratch_dir
        )
    else:
        raise ValueError(f"Unknown render_mode '{render_mode}'. Use 'full' or 'fallback'.")

    for name, stack in (("rgba", rgba_stack), ("depth", depth_stack), ("mask", mask_stack)):
        if stack.shape[0] != NUM_FRAMES:
            raise ValueError(f"{name} has {stack.shape[0]} frames, expected {NUM_FRAMES}.")

    seq_dir = output_root / f"sequence_{sequence_id:04d}"
    _save_sequence_frames(seq_dir, rgba_stack, depth_stack, mask_stack)


def parse_args() -> argparse.Namespace:
    """Parses CLI options for output destination and sequence count."""
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
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=("auto", "full", "fallback"),
        default="fallback",
        help="Rendering strategy: auto-detect, force keyframed full render, or force framewise fallback.",
    )
    return parser.parse_args()


def _choose_render_mode(output_root: Path, scratch_root: Path | None, requested_mode: str) -> str:
    """Chooses fastest safe render path.

    - `full`: force keyframed full-sequence rendering.
    - `fallback`: force robust framewise rendering.
    - `auto`: probe one quick keyframed render once; if no motion is detected,
      switch all subsequent sequences to fallback mode.
    """
    if requested_mode in ("full", "fallback"):
        return requested_mode

    probe_scene = _build_scene(sequence_seed=999, use_keyframes=True)
    probe_scratch = (scratch_root if scratch_root is not None else output_root / "_scratch") / "_probe_mode"
    probe_scratch.mkdir(parents=True, exist_ok=True)
    try:
        rgba_stack, _, _ = _render_full_sequence(probe_scene, scratch_dir=probe_scratch)
        if _has_temporal_motion(rgba_stack):
            return "full"
    except Exception:
        pass
    return "fallback"


if __name__ == "__main__":
    _ = resolve_device(verbose=True)
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    chosen_mode = _choose_render_mode(args.output_root, args.scratch_root, args.render_mode)
    print(f"Using render mode: {chosen_mode}")

    for seq_id in tqdm(range(args.num_sequences), desc="Generating sequences"):
        generate_sequence(
            sequence_id=seq_id,
            output_root=args.output_root,
            scratch_root=args.scratch_root,
            render_mode=chosen_mode,
        )

    print(
        f"Generated {args.num_sequences} sequences in '{args.output_root.resolve()}'. "
        f"Each frame contains rgba_XXX.png, depth_XXX.npy, and mask_XXX.png."
    )
