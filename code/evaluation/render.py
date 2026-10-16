#!/usr/bin/env python3
"""Render SMPL-H motion, or unified SMPL-X motion plus facial expression."""

import argparse
from contextlib import nullcontext
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


SMPLH_LOWER_JOINTS = [0, 1, 3, 4, 6, 7, 9, 10]


def parser():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--jobs-json", help="batch rendering jobs JSON")
    source.add_argument("--npz-path", type=Path, help="single motion NPZ")
    p.add_argument("--wav-path", type=Path, help="optional audio for a single NPZ")
    p.add_argument("--output-path", type=Path, help="output MP4 for a single NPZ")
    p.add_argument("--smplh-model-dir", help="SMPL-H model directory for gesture-only rendering")
    p.add_argument("--smplx-model", type=Path, help="SMPL-X NPZ model; enables whole-person facial animation")
    p.add_argument("--neutral-face", action="store_true", help="SMPL-X comparison with expression, jaw and eyes set to zero")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--concurrent", type=int, default=8)
    p.add_argument("--smpl-batch-size", type=int, default=256)
    p.add_argument("--target-frames", type=int, help="frames to render; defaults to NPZ length for single mode, 600 for batch")
    p.add_argument("--flip-yz", action="store_true")
    p.add_argument("--canonicalize-root", action="store_true")
    p.add_argument("--zero-lower", action="store_true")
    p.add_argument("--zero-global-orient", action="store_true")
    p.add_argument("--center-person", action="store_true")
    p.add_argument("--frame-margin", type=float, default=0.18)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-jobs", type=int)
    p.add_argument("--skip-jobs", type=int, default=0)
    p.add_argument("--system-id")
    p.add_argument("--group-id", action="append")
    p.add_argument("--local-tmp-root", type=Path)
    p.add_argument("--skip-if-mtime-after", type=float)
    p.add_argument("--png-compress-level", type=int)
    return p


def load_pose(
    path, target_frames, device, canonicalize_root=False,
    zero_lower=False, zero_global_orient=False,
):
    """Read pose arrays from an NPZ or an in-memory decoded motion block."""
    base_rotation = None
    with nullcontext(path) if isinstance(path, dict) else np.load(path, allow_pickle=True) as data:
        arrays = {}
        for key in ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose"):
            source_key = key if key in data else f"smplh:{key}"
            value = np.asarray(data[source_key], dtype=np.float32).reshape(len(data[source_key]), -1)
            if key == "global_orient" and canonicalize_root:
                from scipy.spatial.transform import Rotation

                left = min(120, len(value) - 1)
                right = min(480, len(value))
                base_rotation = Rotation.from_rotvec(value[left:right]).mean().as_matrix().astype(np.float32)
            if len(value) < target_frames:
                value = np.concatenate(
                    [value, np.repeat(value[-1:], target_frames - len(value), axis=0)]
                )
            value = value[:target_frames]
            if key == "body_pose" and zero_lower:
                value = value.reshape(target_frames, 21, 3).copy()
                value[:, SMPLH_LOWER_JOINTS] = 0.0
                value = value.reshape(target_frames, 63)
            if key == "global_orient" and zero_global_orient:
                value = np.zeros_like(value)
            arrays[key] = torch.from_numpy(value).to(device)
    return arrays, base_rotation


def load_face(path, target_frames, device, neutral=False):
    """Map 108D facial motion from an NPZ or dictionary to SMPL-X inputs."""
    with nullcontext(path) if isinstance(path, dict) else np.load(path) as data:
        motion = np.asarray(data["motioncode"], dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 108:
        raise ValueError(f"Expected 108D FLAME motioncode, got {motion.shape}")
    if not np.isfinite(motion).all():
        raise ValueError(f"Non-finite FLAME motioncode: {path}")
    if len(motion) < target_frames:
        motion = np.concatenate([motion, np.repeat(motion[-1:], target_frames - len(motion), axis=0)])
    motion = torch.from_numpy(motion[:target_frames].copy()).to(device)
    expression = motion[:, :100]
    jaw = motion.new_zeros(target_frames, 3)
    left_eye = motion.new_zeros(target_frames, 3)
    right_eye = motion.new_zeros(target_frames, 3)
    jaw[:, 0] = motion[:, 103]
    left_eye[:, :2] = motion[:, 104:106]
    right_eye[:, :2] = motion[:, 106:108]
    # Head/neck rotations come from the body stream, avoiding a second head rotation.
    if neutral:
        expression = torch.zeros_like(expression)
        jaw.zero_()
        left_eye.zero_()
        right_eye.zero_()
    return {
        "expression": expression, "jaw_pose": jaw,
        "leye_pose": left_eye, "reye_pose": right_eye,
    }


def decode_vertices(
    model, path, target_frames, batch_size, canonicalize_root=False,
    zero_lower=False, zero_global_orient=False, with_expression=False, neutral_face=False,
):
    """Convert a file or motion dictionary into CPU vertices in frame order."""
    arrays, base_rotation = load_pose(
        path, target_frames, model.shapedirs.device, canonicalize_root,
        zero_lower, zero_global_orient,
    )
    face = load_face(path, target_frames, model.shapedirs.device, neutral_face) if with_expression else {}
    vertices = []
    with torch.inference_mode():
        for left in range(0, target_frames, batch_size):
            right = min(left + batch_size, target_frames)
            output = model(
                betas=torch.zeros((right - left, model.num_betas), device=model.shapedirs.device),
                global_orient=arrays["global_orient"][left:right],
                body_pose=arrays["body_pose"][left:right],
                left_hand_pose=arrays["left_hand_pose"][left:right],
                right_hand_pose=arrays["right_hand_pose"][left:right],
                return_verts=True,
                **{key: value[left:right] for key, value in face.items()},
            )
            vertices.append(output.vertices.cpu())
    vertices = torch.cat(vertices).numpy()
    if base_rotation is not None:
        vertices = vertices @ base_rotation
    return vertices


def install_centered_camera(fast_render, vertices, width, height, margin):
    """Use one safe zoom for the clip while tracking each frame's bbox center."""
    if not 0.0 <= margin < 0.5:
        raise ValueError("--frame-margin must be in [0, 0.5)")
    frame_min = vertices.min(axis=1)
    frame_max = vertices.max(axis=1)
    half_size = (frame_max - frame_min) * 0.5
    aspect = (width // 2) / height
    yfov = float(np.pi / 3)
    tan_half_yfov = float(np.tan(yfov * 0.5))
    content_scale = max(
        float(half_size[:, 1].max()),
        float(half_size[:, 0].max()) / aspect,
    )
    max_half_depth = float(half_size[:, 2].max())
    distance = max_half_depth + content_scale * (1.0 + 2.0 * margin) / tan_half_yfov

    # Conservative bound: combine the largest extent with the closest possible depth.
    closest_depth = distance - max_half_depth
    ndc_x = float(half_size[:, 0].max()) / (closest_depth * tan_half_yfov * aspect)
    ndc_y = float(half_size[:, 1].max()) / (closest_depth * tan_half_yfov)
    minimum_border = (1.0 - max(ndc_x, ndc_y)) * 0.5

    def create_centered_scene(vertices_frame, faces, uniform_color, pose_camera, pose_light):
        trimesh_mesh = fast_render.trimesh.Trimesh(
            vertices=vertices_frame,
            faces=faces,
            vertex_colors=uniform_color,
        )
        mesh = fast_render.pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=True)
        scene = fast_render.pyrender.Scene()
        scene.add(mesh)
        bbox_center = (
            vertices_frame.min(axis=0) + vertices_frame.max(axis=0)
        ) * 0.5
        camera_position = bbox_center + np.asarray([0.0, 0.0, distance])
        camera_pose = fast_render.look_at(camera_position, bbox_center)
        camera = fast_render.pyrender.PerspectiveCamera(yfov=yfov)
        scene.add(camera, pose=camera_pose)
        light = fast_render.pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0], intensity=4.0
        )
        scene.add(light, pose=pose_light)
        return scene

    if not hasattr(fast_render, "look_at"):
        raise RuntimeError("--center-person requires GestureLSM_huggingface fast_render.py")
    fast_render.create_scene_with_mesh = create_centered_scene
    return {
        "camera_distance": distance,
        "conservative_ndc_x": ndc_x,
        "conservative_ndc_y": ndc_y,
        "conservative_minimum_border_fraction": minimum_border,
    }


def render_one(args, model, fast_render, media, job):
    video_path = Path(job["video_path"])
    ready_path = Path(job["ready_path"])
    if (
        args.skip_if_mtime_after is not None
        and video_path.is_file()
        and video_path.stat().st_mtime >= args.skip_if_mtime_after
    ):
        print(f"skip-fixed {video_path}", flush=True)
        return
    if video_path.is_file() and ready_path.is_file() and not args.overwrite:
        print(f"skip {video_path}", flush=True)
        return
    if not Path(job["npz_path"]).is_file():
        raise FileNotFoundError(job["npz_path"])

    tmp_dir = Path(job["tmp_dir"])
    if args.local_tmp_root is not None:
        tmp_dir = args.local_tmp_root / job["system_id"] / job["group_id"]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    vertices = decode_vertices(
        model,
        job["npz_path"],
        args.target_frames,
        args.smpl_batch_size,
        args.canonicalize_root,
        args.zero_lower,
        args.zero_global_orient,
        with_expression=args.smplx_model is not None,
        neutral_face=args.neutral_face,
    )
    if args.flip_yz:
        # GestureLSM-HF's native renderer applies this Rx(pi) coordinate fix.
        vertices[..., 1:3] *= -1
    if args.center_person:
        framing = install_centered_camera(
            fast_render, vertices, args.width, args.height, args.frame_margin
        )
        print(
            "center-framing "
            f"group={job['group_id']} margin={args.frame_margin:.3f} "
            f"distance={framing['camera_distance']:.6f} "
            f"ndc_x={framing['conservative_ndc_x']:.6f} "
            f"ndc_y={framing['conservative_ndc_y']:.6f} "
            f"min_border={framing['conservative_minimum_border_fraction']:.6f}",
            flush=True,
        )
    silent_path = fast_render.generate_silent_videos_no_gt(
        args.fps,
        args.width,
        args.height,
        args.concurrent,
        "png",
        args.target_frames,
        vertices,
        model.faces,
        str(tmp_dir),
    )
    rendered_path = Path(silent_path)
    if job.get("wav_path"):
        rendered_path = tmp_dir / (Path(job["npz_path"]).stem + ".mp4")
        media.add_audio_to_video(silent_path, job["wav_path"], str(rendered_path))
        os.remove(silent_path)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    partial = video_path.with_name(video_path.name + ".partial")
    shutil.copy2(rendered_path, partial)
    os.replace(partial, video_path)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text("ready\n", encoding="utf-8")
    shutil.rmtree(tmp_dir)
    print(video_path, flush=True)


def main():
    arg_parser = parser()
    args = arg_parser.parse_args()
    if args.smplx_model is None and not args.smplh_model_dir:
        arg_parser.error("provide --smplh-model-dir or --smplx-model")
    if args.neutral_face and args.smplx_model is None:
        arg_parser.error("--neutral-face requires --smplx-model")
    if args.npz_path:
        if args.output_path is None:
            arg_parser.error("--npz-path requires --output-path")
        output = args.output_path.resolve()
        jobs = [{
            "npz_path": str(args.npz_path.resolve()),
            "wav_path": str(args.wav_path.resolve()) if args.wav_path else "",
            "video_path": str(output),
            "ready_path": str(output.with_suffix(".ready")),
            "tmp_dir": str(output.parent / (output.stem + ".render_tmp")),
            "system_id": "single", "group_id": args.npz_path.stem,
        }]
        if args.target_frames is None:
            with np.load(args.npz_path) as data:
                key = "body_pose" if "body_pose" in data else "smplh:body_pose"
                args.target_frames = len(data[key])
    else:
        if args.wav_path or args.output_path:
            arg_parser.error("--wav-path and --output-path require --npz-path")
        jobs = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))
        if args.target_frames is None:
            args.target_frames = 600
    if args.flip_yz and args.canonicalize_root:
        raise ValueError("--flip-yz and --canonicalize-root are mutually exclusive")
    if args.zero_global_orient and args.canonicalize_root:
        raise ValueError("--zero-global-orient and --canonicalize-root are mutually exclusive")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYOPENGL_LOG_LEVEL", "ERROR")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import smplx
    from utils import fast_render, media

    if args.png_compress_level is not None:
        original_imwrite = fast_render.imageio.imwrite

        def fast_png_imwrite(path, image, *write_args, **write_kwargs):
            write_kwargs["compress_level"] = args.png_compress_level
            return original_imwrite(path, image, *write_args, **write_kwargs)

        fast_render.imageio.imwrite = fast_png_imwrite

    if args.smplx_model is not None:
        model = smplx.SMPLX(
            str(args.smplx_model), gender="neutral", ext="npz",
            num_expression_coeffs=100, use_pca=False, flat_hand_mean=True,
        ).to(args.device).eval()
        if model.num_expression_coeffs != 100:
            raise ValueError("The SMPL-X asset must contain the full 100D expression space")
    else:
        model = smplx.create(
            args.smplh_model_dir, model_type="smplh", gender="FEMALE", ext="pkl",
            use_pca=False, flat_hand_mean=True,
        ).to(args.device).eval()
    if args.system_id is not None:
        jobs = [job for job in jobs if job["system_id"] == args.system_id]
    if args.group_id:
        group_ids = set(args.group_id)
        jobs = [job for job in jobs if job["group_id"] in group_ids]
    jobs = jobs[args.skip_jobs :]
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    for job in jobs:
        render_one(args, model, fast_render, media, job)


if __name__ == "__main__":
    main()
