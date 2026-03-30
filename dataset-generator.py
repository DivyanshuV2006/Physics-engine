from __future__ import annotations

import argparse
import shutil
import sys
import types
from pathlib import Path
from typing import Dict

import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

"""Kubric synthetic data generator for occlusion-centric video sequences.

This script intentionally avoids PyBullet simulation and drives the target
sphere with explicit transform keyframes so render-time is low and motion is
deterministic.
"""

# Kubric 0.1.x expects `tfds.core.ReadWritePath`, which is absent in some newer
# tensorflow-datasets releases where `tfds.core.Path` is used instead.
try:
    import tensorflow_datasets as tfds

    if not hasattr(tfds.core, "ReadWritePath") and hasattr(tfds.core, "Path"):
        tfds.core.ReadWritePath = tfds.core.Path  # type: ignore[attr-defined]
    if "tensorflow_datasets.core.utils.generic_path" not in sys.modules:
        generic_path_mod = types.ModuleType("tensorflow_datasets.core.utils.generic_path")
        generic_path_mod.as_path = tfds.core.as_path
        sys.modules["tensorflow_datasets.core.utils.generic_path"] = generic_path_mod
except Exception:
    # Keep import-time behavior unchanged if TFDS is unavailable for any reason.
    pass

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

TARGET_X_START = -4.0
TARGET_X_END = 4.0

SEMANTIC_ID_FLOOR = 1
SEMANTIC_ID_WALL = 2
SEMANTIC_ID_TARGET = 3


def _build_scene(sequence_seed: int = 0) -> tuple[kb.Scene, kb.Sphere]:
    """Creates one scene with a manually keyframed kinematic target sphere."""
    rng = np.random.default_rng(sequence_seed)
    scene = kb.Scene(
        resolution=RESOLUTION,
        frame_start=FRAME_START,
        frame_end=FRAME_END,
        frame_rate=FPS,
    )

    scene.camera = kb.PerspectiveCamera(
        name="camera",
        position=(0.0, -10.0, 4.0),
        look_at=(0.0, 0.0, 0.75),
    )
    scene += scene.camera

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
        material=kb.FlatMaterial(color=(0.75, 0.75, 0.75)),
    )
    floor.semantic_id = SEMANTIC_ID_FLOOR
    floor.segmentation_id = SEMANTIC_ID_FLOOR
    scene += floor

    wall = kb.Cube(
        name="wall_occluder",
        scale=(0.65, 1.8, 1.8),
        position=(0.0, -1.2, 1.0),
        static=True,
        material=kb.FlatMaterial(color=(0.90, 0.05, 0.05)),
    )
    wall.semantic_id = SEMANTIC_ID_WALL
    wall.segmentation_id = SEMANTIC_ID_WALL
    scene += wall

    # Keep the target static/kinematic because we bypass the PyBullet simulator.
    target_y = float(rng.uniform(0.6, 1.2))
    target = kb.Sphere(
        name="target",
        scale=0.45,
        position=(TARGET_X_START, target_y, 0.45),
        static=True,
        material=kb.FlatMaterial(color=(0.05, 0.25, 0.95)),
    )
    target.semantic_id = SEMANTIC_ID_TARGET
    target.segmentation_id = SEMANTIC_ID_TARGET
    scene += target

    return scene, target


def _apply_target_motion_keyframes(target: kb.Sphere) -> None:
    """Applies explicit per-frame keyframes after renderer observers are attached."""
    target_y = float(target.position[1])
    target_z = float(target.position[2])
    frame_ids = np.arange(FRAME_START, FRAME_END + 1, dtype=np.int32)
    x_positions = np.linspace(TARGET_X_START, TARGET_X_END, num=frame_ids.size)
    for frame_id, x_pos in zip(frame_ids, x_positions):
        target.position = (float(x_pos), target_y, target_z)
        target.keyframe_insert("position", int(frame_id))


def _select_segmentation_map(render_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Retrieves segmentation output from renderer results across API variants."""
    if "semantic_segmentation" in render_data:
        return render_data["semantic_segmentation"]
    if "segmentation" in render_data:
        return render_data["segmentation"]
    raise KeyError(
        "No segmentation output found in renderer data. "
        "Expected one of: semantic_segmentation, segmentation."
    )


def _validate_render_frame_counts(
    rgba_stack: np.ndarray,
    depth_stack: np.ndarray,
    segmentation_stack: np.ndarray,
) -> None:
    """Ensures all renderer outputs have the expected frame count."""
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


def _resolve_target_label(segmentation_stack: np.ndarray) -> tuple[int, float]:
    """Finds the segmentation label that corresponds to the moving target.

    Works across Kubric variants where segmentation may be semantic IDs or
    per-instance IDs.
    """
    seg_stack = np.squeeze(np.asarray(segmentation_stack)).astype(np.int32)
    if seg_stack.ndim != 3:
        raise ValueError(f"Unexpected segmentation stack shape: {seg_stack.shape}")

    def _measure_shift(label_id: int) -> tuple[float, int]:
        x_centers = []
        max_area = 0
        for seg in seg_stack:
            ys, xs = np.where(seg == label_id)
            if xs.size > 0:
                x_centers.append(float(np.mean(xs)))
                max_area = max(max_area, int(xs.size))
        if len(x_centers) < 2:
            return 0.0, max_area
        return float(x_centers[-1] - x_centers[0]), max_area

    # Prefer declared semantic target ID when it actually moves.
    semantic_shift, semantic_area = _measure_shift(SEMANTIC_ID_TARGET)
    if abs(semantic_shift) > 5.0 and semantic_area > 0:
        return SEMANTIC_ID_TARGET, semantic_shift

    # Fallback: pick the non-background label with the strongest horizontal motion.
    candidates = [int(i) for i in np.unique(seg_stack) if int(i) != 0]
    if not candidates:
        raise RuntimeError("No non-background labels found in segmentation output.")

    best_id = None
    best_shift = 0.0
    best_area = None
    for label_id in candidates:
        shift, area = _measure_shift(label_id)
        if best_id is None:
            best_id, best_shift, best_area = label_id, shift, area
            continue
        # Prefer larger motion; on ties prefer smaller object footprint.
        if abs(shift) > abs(best_shift) or (abs(shift) == abs(best_shift) and area < best_area):
            best_id, best_shift, best_area = label_id, shift, area

    if best_id is None or abs(best_shift) <= 5.0:
        raise RuntimeError(
            "Frozen target detected: no segmentation label shows sufficient horizontal motion."
        )
    return best_id, best_shift


def _save_sequence_frames(
    sequence_dir: Path,
    rgba_stack: np.ndarray,
    depth_stack: np.ndarray,
    segmentation_stack: np.ndarray,
    target_label: int,
) -> None:
    """Writes per-frame RGB PNG, depth NPY, and binary target mask PNG."""
    sequence_dir.mkdir(parents=True, exist_ok=True)

    num_frames = rgba_stack.shape[0]
    for f in range(num_frames):
        rgba = np.asarray(rgba_stack[f])
        if rgba.ndim != 3:
            raise ValueError(f"Unexpected rgba shape at frame {f}: {rgba.shape}")
        rgb = rgba[..., :3] if rgba.shape[-1] >= 3 else rgba
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        depth = np.asarray(depth_stack[f], dtype=np.float32)
        depth = np.squeeze(depth).astype(np.float32)

        seg = np.asarray(segmentation_stack[f])
        seg = np.squeeze(seg).astype(np.int32)
        mask = (seg == target_label).astype(np.uint8) * 255

        iio.imwrite(sequence_dir / f"rgba_{f:03d}.png", rgb)
        np.save(sequence_dir / f"depth_{f:03d}.npy", depth)
        iio.imwrite(sequence_dir / f"mask_{f:03d}.png", mask)

    # Hard guarantee: final sequence folders keep only training artifacts.
    for exr_file in sequence_dir.glob("*.exr"):
        exr_file.unlink(missing_ok=True)


def generate_sequence(
    sequence_id: int,
    output_root: Path,
    scratch_root: Path | None = None,
    keep_scratch: bool = False,
) -> None:
    """Renders one sequence and exports validated frame artifacts."""
    scene, target = _build_scene(sequence_seed=sequence_id)

    if scratch_root is None:
        scratch_base = output_root / "_scratch"
    else:
        scratch_base = scratch_root
    scratch_base.mkdir(parents=True, exist_ok=True)
    scratch_dir = scratch_base / f"sequence_{sequence_id:04d}"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    renderer = Blender(scene, scratch_dir=scratch_dir, use_denoising=False)
    _apply_target_motion_keyframes(target)
    render_data = renderer.render()
    if render_data is None:
        # Kubric 0.1.x writes renders to disk and exposes arrays via postprocess.
        render_data = renderer.postprocess(from_dir=scratch_dir, to_dir=scratch_dir)

    if "rgba" not in render_data or "depth" not in render_data:
        missing = [k for k in ("rgba", "depth") if k not in render_data]
        raise KeyError(f"Missing renderer outputs: {missing}")

    rgba_stack = np.asarray(render_data["rgba"])
    depth_stack = np.asarray(render_data["depth"])
    segmentation_stack = np.asarray(_select_segmentation_map(render_data))

    _validate_render_frame_counts(rgba_stack, depth_stack, segmentation_stack)
    target_label, _ = _resolve_target_label(segmentation_stack)

    seq_dir = output_root / f"sequence_{sequence_id:04d}"
    _save_sequence_frames(seq_dir, rgba_stack, depth_stack, segmentation_stack, target_label)

    if not keep_scratch:
        shutil.rmtree(scratch_dir, ignore_errors=True)


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
        help="Optional root directory for Kubric/Blender scratch files.",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep renderer scratch files (default deletes per-sequence scratch).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for seq_id in tqdm(range(args.num_sequences), desc="Generating sequences"):
        generate_sequence(
            sequence_id=seq_id,
            output_root=args.output_root,
            scratch_root=args.scratch_root,
            keep_scratch=args.keep_scratch,
        )

    print(
        f"Generated {args.num_sequences} sequences in '{args.output_root.resolve()}'. "
        "Each frame contains rgba_XXX.png, depth_XXX.npy, and mask_XXX.png."
    )
