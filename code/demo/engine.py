"""Resident models with overlapping token generation, RVQ decoding and rendering."""
from contextlib import closing
from copy import copy
from pathlib import Path
import json
import time

import numpy as np
import torch

from infer import RELEASE_ARGS
from scripts import infer_ellsa2_gesture_expression_from_audio as inference
from demo.render import Renderer
from demo.smoothing import smooth_boundary
from demo.prefixes import DEFAULT_PREFIX, load_motion_prefixes


class Engine:
    def __init__(self, options):
        """Load models and warm rendering workers once for all requests."""
        self.args = inference.finalize_args(inference.parse_args([
            *RELEASE_ARGS, *(["--motion-bias"] if options.motion_bias else []),
            "--motion-style-pose-path", str(options.style_path),
            "--device", options.device,
            "--rvq-decode-mode", ("whole" if options.decode_window_blocks == 0
                                  and getattr(options, "rvq_decode_mode", "auto") == "auto"
                                  else getattr(options, "rvq_decode_mode", "auto")),
        ]))
        self.decode_window_blocks = options.decode_window_blocks
        self.boundary_smoothing_frames = getattr(options, "boundary_smoothing_frames", 0)
        device = inference.choose_device(options.device)
        self.tokenizer, self.model = inference.load_ellsa2_model(
            self.args, device, inference.choose_dtype(self.args.dtype, device))
        self.decoders = inference.load_motion_decoders(self.args, device)
        self.stats = inference.load_gesture_stats(
            self.args.stats_dir, self.args.stats_prefix, self.args.stats_suffix)
        self.flame = inference.load_flame_decoder(
            self.args.vqvae_flame_path, device, inference.resolve_rvq_context_frames(self.args, "expression"),
            self.args.rvq_encoder_dtype, self.args.rvq_quantizer_distance_dtype)
        self.mean, self.std = inference.load_flame_stats(self.args.flame_stats_path)
        if inference.use_streaming_rvq(self.decoders, self.flame, self.args):
            self.decode_window_blocks = 1
        self.fbank = inference.Fbank(inference.FbankConfig(num_mel_bins=128))
        self.style = inference.encode_motion_style_tokens(
            str(options.style_path), self.args.motion_style_flame_path, 0, 612,
            self.decoders, self.stats, self.flame, self.mean, self.std, self.args)
        self.style = self.style[:len(self.style) // 6 * 6]
        self.prefixes = load_motion_prefixes()
        self.prefix_tokens = {}
        for prefix_id, prefix in self.prefixes.items():
            if prefix["pose_path"].resolve() == Path(options.style_path).resolve():
                tokens = self.style
            else:
                tokens = inference.encode_motion_style_tokens(
                    str(prefix["pose_path"]), "", 0, prefix["source_frames"], self.decoders,
                    self.stats, self.flame, self.mean, self.std, self.args)
                tokens = tokens[:len(tokens) // self.args.motion_steps_per_window * self.args.motion_steps_per_window]
            self.prefix_tokens[prefix_id] = tokens
        self.renderer = Renderer(options.smplx_model, device)

    def iter_tokens(self, target, partner, windows=None, *, args=None):
        """Each yield is one 0.8 s block of 6 × 18 discrete motion tokens."""
        args = copy(self.args) if args is None else args
        args.num_windows = windows or (max(len(target), len(partner)) + 12799) // 12800
        yield from inference.iter_gesture_expression_tokens(
            target, partner, self.tokenizer, self.model, self.fbank, args,
            speaker_motion_ids=self.style, motion_decoders=self.decoders, gesture_stats=self.stats)

    def _iter_motion(self, token_stream, chunks, args, log, start):
        """Commit prefixes once; window=1 uses no subsequent audio/motion blocks."""
        if inference.use_streaming_rvq(self.decoders, self.flame, args):
            args.rvq_active_decode_mode = "stream"
            decoder = inference.StreamingMotionDecoder(
                self.decoders, self.stats, self.flame, self.mean, self.std, args)
            with closing(token_stream):
                for tokens in token_stream:
                    chunks.append(tokens)
                    poses, face = decoder.decode(tokens)
                    yield {**poses, "motioncode": face}
            log.write(f"Token generation complete: {time.perf_counter() - start:.3f} s\n")
            log.flush()
            return
        args.rvq_active_decode_mode = "whole"
        window = self.decode_window_blocks
        overlap = min(2, max(0, window - 1))
        decode_at, committed, decoded_count = window, 0, 0
        decoded = None
        with closing(token_stream):
            for tokens in token_stream:
                chunks.append(tokens)
                if not window or len(chunks) < decode_at:
                    continue
                decoded = inference.decode_tokens(torch.cat(chunks), self.decoders, self.stats,
                                                  self.flame, self.mean, self.std, args)
                decoded_count = len(chunks)
                right = (len(chunks) - overlap) * 24
                poses, face = decoded
                yield {key: value[committed:right].copy()
                       for key, value in {**poses, "motioncode": face}.items()}
                committed = right
                decode_at += window - overlap
        log.write(f"Token generation complete: {time.perf_counter() - start:.3f} s\n")
        log.flush()
        # Flush the held lookahead and any incomplete final window without padding.
        if decoded_count != len(chunks):
            decoded = inference.decode_tokens(torch.cat(chunks), self.decoders, self.stats,
                                              self.flame, self.mean, self.std, args)
        poses, face = decoded
        if committed < len(face):
            yield {key: value[committed:].copy()
                   for key, value in {**poses, "motioncode": face}.items()}

    def generate(self, directory, windows, pose_guide=None, guide_time=3.2, guide_start=0):
        """Stream video and save the same committed poses used for its frames."""
        directory = Path(directory)
        target = inference.load_audio(str(directory / "target.wav"), 16000)
        partner = inference.load_audio(str(directory / "partner.wav"), 16000)
        args = copy(self.args)
        if pose_guide and not args.motion_bias:
            raise ValueError("Start the demo with --motion-bias to use a pose guide.")
        args.pose_guide, args.pose_guide_time, args.pose_guide_start = pose_guide or "", guide_time, guide_start
        inference.finalize_args(args)
        args.decode_window_blocks = self.decode_window_blocks
        args.boundary_smoothing_frames = getattr(self, "boundary_smoothing_frames", 0)
        tokens = self.iter_tokens(target, partner, windows, args=args)
        yield from self._render_tokens(directory, tokens, args, windows, directory / "dialogue.wav")

    def generate_microphone(self, audio, prefix_id=DEFAULT_PREFIX):
        """Consume incoming windows without knowing the recording's final length."""
        args = copy(self.args)
        if not inference.use_streaming_rvq(self.decoders, self.flame, args):
            raise ValueError("Live microphone generation requires the released causal RVQ decoders.")
        prefix = self.prefixes[prefix_id]
        args.motion_style_pose_path = str(prefix["pose_path"])
        args.motion_style_start = 0
        args.motion_style_length = prefix["source_frames"]
        prefix_file = audio.directory / "motion_prefix.json"
        prefix_file.write_text(json.dumps(
            {key: value for key, value in prefix.items() if key != "pose_path"}, indent=2) + "\n")
        args.decode_window_blocks = 1
        args.boundary_smoothing_frames = getattr(self, "boundary_smoothing_frames", 0)
        tokens = inference.iter_gesture_expression_tokens(
            None, None, self.tokenizer, self.model, self.fbank, args,
            speaker_motion_ids=self.prefix_tokens[prefix_id], audio_windows=iter(audio))
        for video, files, status in self._render_tokens(audio.directory, tokens, args, None, None):
            if files:
                files.append(str(prefix_file))
            yield video, files, status

    def _render_tokens(self, directory, tokens, args, windows, audio_path):
        """Use one continuous encoder for upload playback or muted live playback."""
        start = time.perf_counter()
        motion_path = directory / "motion.npz"
        chunks, motion_blocks = [], []
        yield None, None, "Listening…" if windows is None else "Generating motion and rendering video…"
        with (directory / "run.log").open("a") as log:
            def vertices():
                """Advance generation while previously submitted frames rasterize."""
                previous = None
                with closing(self._iter_motion(tokens, chunks, args, log, start)) as blocks:
                    for block in blocks:
                        smooth_boundary(block, previous, args.boundary_smoothing_frames)
                        previous = block
                        motion_blocks.append(block)
                        log.write(f"Committed {sum(len(b['body_pose']) for b in motion_blocks)} frames: "
                                  f"{time.perf_counter() - start:.3f} s\n")
                        log.flush()
                        yield self.renderer.vertices(block)
                combined = {key: np.concatenate([block[key] for block in motion_blocks])
                            for key in motion_blocks[0]}
                face = combined.pop("motioncode")
                inference.decode_and_save_tokens(
                    torch.cat(chunks), str(motion_path), self.decoders, self.stats,
                    self.flame, self.mean, self.std, args, decoded_motion=(combined, face))
                log.write(f"Motion saved: {time.perf_counter() - start:.3f} s\n")
                log.flush()

            with closing(self.renderer.stream_vertices(
                    vertices(), audio_path, directory, log,
                    segment_frames=24 if args.decode_window_blocks == 1 else 48,
                    prefetch_batches=1 if windows is None else 4)) as segments:
                for index, segment in enumerate(segments):
                    log.write(f"Segment {index}: {time.perf_counter() - start:.3f} s\n")
                    log.flush()
                    count = f"{len(chunks)}/{windows}" if windows is not None else str(len(chunks))
                    yield str(segment), None, f"Playing · {count} motion blocks generated"
            if windows is None:
                # Keep the live preview muted; add the recorded voice to the download.
                import subprocess
                silent = directory / "silent.mp4"
                (directory / "video.mp4").rename(silent)
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(silent),
                                "-i", str(directory / "dialogue.wav"), "-map", "0:v", "-map", "1:a",
                                "-c:v", "copy", "-c:a", "aac", "-shortest", "-movflags", "+faststart",
                                str(directory / "video.mp4")], check=True, stdout=log, stderr=log)
        count = len(chunks)
        files = [str(motion_path), str(directory / "video.mp4")]
        if windows is None:
            files.append(str(directory / "target.wav"))
        yield None, files, f"Done · {count * 0.8:g} s · {count * 24} frames"

    def compare(self, directory):
        """Render the saved tokens in full, then compare both decodings in sync."""
        from demo.comparison import compare_motion, compose_video, comparison_files
        import json

        directory = Path(directory)
        paired = directory / "comparison.mp4"
        if paired.is_file() and (directory / "comparison.json").is_file():
            yield str(paired), comparison_files(directory), "Comparison ready · same generated tokens"
            return
        whole = directory / "whole"
        whole.mkdir(exist_ok=True)
        with np.load(directory / "motion.npz") as motion:
            tokens = torch.from_numpy(motion["rvq_combined_code_idx"].copy())
            window = int(motion["rvq_decode_window_blocks"])
            smoothing = int(motion.get("motion_boundary_smoothing_frames", 0))
        args = copy(self.args)
        args.decode_window_blocks = 0
        args.rvq_active_decode_mode = "whole"
        args.boundary_smoothing_frames = 0
        args.inline_exemplar_enabled = False
        inference.decode_and_save_tokens(tokens, str(whole / "motion.npz"), self.decoders,
                                         self.stats, self.flame, self.mean, self.std, args)
        metrics = compare_motion(directory / "motion.npz", whole / "motion.npz")
        with (directory / "comparison.log").open("a") as log:
            for index, _ in enumerate(self.renderer.stream(
                    whole / "motion.npz", directory / "dialogue.wav", whole, log, segment_frames=24)):
                yield None, None, f"Rendering whole-sequence reference · {index + 1} segments"
            with np.load(directory / "motion.npz") as left, np.load(whole / "motion.npz") as right:
                distance = np.linalg.norm(self.renderer.vertices(dict(left)) - self.renderer.vertices(dict(right)), axis=-1) * 1000
            metrics.update(vertex_mean_mm=float(distance.mean()), vertex_p95_mm=float(np.quantile(distance, 0.95)),
                           vertex_max_mm=float(distance.max()))
            label = "Zero lookahead (0.8 s)" if window == 1 else ("Whole sequence" if window == 0 else f"Buffered ({window * 0.8:g} s)")
            if window == 1 and smoothing:
                label = "Smoothed zero lookahead"
            compose_video(directory / "video.mp4", whole / "video.mp4", paired, log, label)
        (directory / "comparison.json").write_text(json.dumps(metrics, indent=2))
        yield str(paired), comparison_files(directory), "Comparison ready · same generated tokens"

    def close(self):
        """Release the persistent rendering processes when the server stops."""
        self.renderer.close()
