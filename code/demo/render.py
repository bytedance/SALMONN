"""Persistent rendering workers and one continuous video/audio encoder."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess

import numpy as np


def _init_worker(faces, rest_vertices):
    """Keep one EGL context, mesh topology, camera and light per worker."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("LP_NUM_THREADS", "2")
    from utils import fast_render
    from OpenGL.GL import glGetString, GL_RENDERER

    global _render, _mesh, _scene, _node
    _render = fast_render.pyrender.OffscreenRenderer(480, 720)
    _mesh = fast_render.trimesh.Trimesh(vertices=rest_vertices, faces=faces,
                                      vertex_colors=[220, 220, 220, 255], process=False)
    _scene = fast_render.pyrender.Scene()
    _node = _scene.add(fast_render.pyrender.Mesh.from_trimesh(_mesh, smooth=True))
    _scene.add(fast_render.pyrender.OrthographicCamera(xmag=1.0, ymag=1.0),
               pose=fast_render.create_pose_camera(-4))
    _scene.add(fast_render.pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0),
               pose=fast_render.create_pose_light(-30))
    _render.render(_scene)
    print(f"EGL renderer: {glGetString(GL_RENDERER).decode()}", flush=True)


def _frames(vertices):
    """Update deformed geometry while reusing the scene and fixed mesh topology."""
    from utils import fast_render

    global _node
    images = []
    for frame in vertices:
        # SMPL-X already has a valid, fixed topology; no vertex merging is needed.
        _mesh.vertices = frame
        _scene.remove_node(_node)
        _node = _scene.add(fast_render.pyrender.Mesh.from_trimesh(_mesh, smooth=True))
        images.append(_render.render(_scene)[0])
    return np.stack(images)


class Renderer:
    def __init__(self, model_path, device, workers=2):
        """Load SMPL-X and start persistent rasterization workers."""
        import smplx
        import torch

        self.model = smplx.SMPLX(str(model_path), gender="neutral", ext="npz",
                                num_expression_coeffs=100, use_pca=False, flat_hand_mean=True).to(device).eval()
        if self.model.num_expression_coeffs != 100:
            raise ValueError("The SMPL-X model must contain 100 expression coefficients")
        with torch.inference_mode():
            rest = self.model().vertices[0].cpu().numpy()
        self.pool = ProcessPoolExecutor(workers, mp_context=get_context("spawn"),
                                        initializer=_init_worker, initargs=(self.model.faces, rest))
        # Start the workers now rather than during the first video.
        list(self.pool.map(_frames, [rest[None]] * workers))

    def vertices(self, motion):
        """Convert an in-memory motion block to vertices using the CLI path."""
        from evaluation.render import decode_vertices

        return decode_vertices(self.model, motion, len(motion["body_pose"]), 256,
                               zero_lower=True, zero_global_orient=True, with_expression=True)

    def stream(self, motion_path, audio_path, directory, log, segment_frames=48):
        """Render an already saved NPZ with the same continuous encoder."""
        with np.load(motion_path) as motion:
            vertices = self.vertices(dict(motion))
        yield from self.stream_vertices(iter([vertices]), audio_path, directory, log, segment_frames)

    def stream_vertices(self, vertex_blocks, audio_path, directory, log, segment_frames=48, prefetch_batches=4):
        """Consume lazy GPU-generated blocks while CPU rendering runs in parallel."""
        directory = Path(directory)
        segments = directory / "segments"
        segments.mkdir()
        # One encoder keeps timestamps and AAC state continuous across segments.
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-video_size", "480x720", "-framerate", "30", "-i", "pipe:0",
            *(["-i", str(audio_path)] if audio_path is not None else []),
            "-map", "0:v", *(["-map", "1:a"] if audio_path is not None else []),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-tune", "zerolatency", "-threads", "2",
            "-g", str(segment_frames), "-keyint_min", str(segment_frames), "-sc_threshold", "0",
            *(["-c:a", "aac", "-b:a", "64k", "-shortest"] if audio_path is not None else []),
            "-flags", "+global_header", "-f", "tee",
            "[f=mp4:movflags=+faststart]video.mp4|"
            f"[f=hls:hls_time={segment_frames / 30}:hls_list_size=0:"
            "hls_flags=independent_segments+temp_file:"
            "hls_segment_filename=segments/%05d.ts]segments/stream.m3u8",
        ]
        process = subprocess.Popen(command, cwd=directory, stdin=subprocess.PIPE,
                                   stdout=log, stderr=log)
        emitted = 0
        # Resolve futures in order with a bounded number of batches in flight.
        batches = (block[start:start + 12] for block in vertex_blocks
                   for start in range(0, len(block), 12))
        pending = deque()
        try:
            pending.extend(self.pool.submit(_frames, batch) for batch in islice(batches, prefetch_batches))
            while pending:
                frames = pending.popleft().result()
                process.stdin.write(frames.tobytes())
                process.stdin.flush()
                while (path := segments / f"{emitted:05d}.ts").exists():
                    yield path
                    emitted += 1
                # Publish ready frames before generating another block.
                if (batch := next(batches, None)) is not None:
                    pending.append(self.pool.submit(_frames, batch))
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("FFmpeg failed; inspect this request's run.log")
            while (path := segments / f"{emitted:05d}.ts").exists():
                yield path
                emitted += 1
        finally:
            if hasattr(vertex_blocks, "close"):
                vertex_blocks.close()
            for future in pending:
                future.cancel()
            if process.poll() is None:
                process.terminate()
                process.wait()
            if not process.stdin.closed:
                process.stdin.close()

    def close(self):
        """Finish pending work and release the EGL worker processes."""
        self.pool.shutdown()
