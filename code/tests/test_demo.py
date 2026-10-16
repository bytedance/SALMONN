"""Exercise the demo's real audio conversion without loading model weights."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np
import soundfile as sf

from demo.app import generate, prepare_audio


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class DemoAudioTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "stereo.flac"
        self.partner = self.root / "short.wav"
        tone = np.sin(2 * np.pi * 220 * np.arange(96000) / 48000) * 0.25
        sf.write(self.target, np.column_stack((tone, tone)), 48000)
        sf.write(self.partner, np.full(17600, 0.125), 16000)

    def test_conversion_preserves_both_timelines_and_pads_last_block(self):
        with (self.root / "test.log").open("w") as log:
            windows = prepare_audio(self.target, self.partner, self.root, 60, log)
        self.assertEqual(windows, 3)
        stereo, rate = sf.read(self.root / "dialogue.wav")
        self.assertEqual((rate, stereo.shape), (16000, (38400, 2)))
        target, _ = sf.read(self.root / "target.wav")
        partner, _ = sf.read(self.root / "partner.wav")
        np.testing.assert_array_equal(stereo[:32000, 0], target)
        np.testing.assert_array_equal(stereo[:17600, 1], partner)
        self.assertTrue(np.all(stereo[32000:, 0] == 0))
        self.assertTrue(np.all(stereo[17600:, 1] == 0))
        with wave.open(str(self.root / "target.wav")) as audio:
            self.assertEqual((audio.getnchannels(), audio.getnframes()), (1, 32000))

    def test_duration_limit_is_an_error_not_truncation(self):
        with (self.root / "test.log").open("w") as log:
            with self.assertRaisesRegex(ValueError, "at most 1 seconds"):
                prepare_audio(self.target, self.partner, self.root, 1, log)
        self.assertFalse((self.root / "dialogue.wav").exists())

    def test_missing_upload_does_not_create_a_job(self):
        output = self.root / "outputs"
        with self.assertRaisesRegex(ValueError, "Upload both"):
            next(generate(str(self.target), None, SimpleNamespace(output_dir=output)))
        self.assertFalse(output.exists())


class DemoMotionTest(unittest.TestCase):
    """Check that streamed frames and downloaded motion share committed decoding."""

    def setUp(self):
        """Replace expensive models with a decoder whose past depends on the future."""
        import torch
        from demo.engine import Engine, inference
        from infer import RELEASE_ARGS

        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = Engine.__new__(Engine)
        style = self.root / "style.npz"
        np.savez(style, body_pose=np.zeros((612, 63), dtype=np.float32))
        self.engine.args = inference.finalize_args(inference.parse_args([
            *RELEASE_ARGS, "--motion-style-pose-path", str(style)]))
        self.engine.decoders, self.engine.stats = {}, {}
        self.engine.flame, self.engine.mean, self.engine.std = None, None, None
        self.generated, self.rendered, self.closed = [], [], []

        def tokens(target, partner, windows, *, args):
            """Expose how far generation has advanced when each segment arrives."""
            try:
                for index in range(windows):
                    self.generated.append(index)
                    yield torch.full((6, 18), index, dtype=torch.long)
            finally:
                self.closed.append(True)

        def decode(token_ids, *unused):
            """Make prefix decoding differ from decoding the same tokens in full."""
            value = np.repeat(token_ids[:, 0].numpy(), 4).astype(np.float32)
            value += len(token_ids) / 1000
            poses = {key: np.repeat(value[:, None], dim, axis=1)
                     for key, dim in (("body_pose", 63), ("left_hand_pose", 45),
                                      ("right_hand_pose", 45), ("global_orient", 3),
                                      ("translation", 3))}
            return poses, np.repeat(value[:, None], 108, axis=1)

        def vertices(block):
            """Remember the exact pose arrays supplied to rendering."""
            self.rendered.append({key: value.copy() for key, value in block.items()})
            return block["body_pose"][:, None, :3]

        def stream(blocks, audio, directory, log, **options):
            """Emit segments lazily, closing upstream generation on cancellation."""
            try:
                for index, block in enumerate(blocks):
                    yield directory / f"{index}.ts"
            finally:
                blocks.close()

        self.engine.iter_tokens = tokens
        self.engine.renderer = SimpleNamespace(vertices=vertices, stream_vertices=stream)
        self.decode = decode
        for name, replacement in (("decode_tokens", decode), ("load_audio", lambda *args: np.zeros(12800))):
            patcher = patch.object(inference, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_first_segment_precedes_generation_end_and_matches_download(self):
        """Check useful early playback and identical saved body/hand/face arrays."""
        self.engine.decode_window_blocks = 8
        results = self.engine.generate(self.root, 25)
        next(results)
        first = next(results)
        self.assertIsNotNone(first[0])
        self.assertEqual(len(self.generated), 8)
        list(results)
        with np.load(self.root / "motion.npz") as saved:
            for key in self.rendered[0]:
                np.testing.assert_array_equal(saved[key], np.concatenate([b[key] for b in self.rendered]))
            self.assertEqual(saved["body_pose"].shape, (600, 63))
            self.assertEqual(saved["rvq_decode_window_blocks"], 8)
            self.assertEqual(saved["rvq_combined_code_idx"].shape, (150, 18))
        self.assertEqual(len(self.generated), 25)

    def test_zero_lookahead_commits_each_block_without_later_tokens(self):
        """Every streamed block must equal decoding its own available prefix."""
        self.engine.decode_window_blocks = 1
        results = self.engine.generate(self.root, 5)
        next(results)
        next(results)
        self.assertEqual(len(self.generated), 1)
        first = self.rendered[0]["body_pose"].copy()
        list(results)
        with np.load(self.root / "motion.npz") as saved:
            tokens = saved["rvq_combined_code_idx"]
            for block in range(5):
                poses, face = self.decode(self.torch.from_numpy(tokens[:(block + 1) * 6]))
                np.testing.assert_array_equal(saved["body_pose"][block * 24:(block + 1) * 24], poses["body_pose"][-24:])
                np.testing.assert_array_equal(saved["motioncode"][block * 24:(block + 1) * 24], face[-24:])
            np.testing.assert_array_equal(first, saved["body_pose"][:24])
        self.assertTrue(all(len(block["body_pose"]) == 24 for block in self.rendered))

    def test_short_windows_and_tail_keep_every_frame_once(self):
        """Cover short clips, exact boundaries and held future-context tails."""
        for window in (0, 1, 2, 8):
            for count in (1, 7, 8, 9, 14, 25):
                with self.subTest(window=window, count=count):
                    directory = self.root / f"{window}_{count}"
                    directory.mkdir()
                    self.generated.clear()
                    self.rendered.clear()
                    self.engine.decode_window_blocks = window
                    list(self.engine.generate(directory, count))
                    with np.load(directory / "motion.npz") as saved:
                        tokens = saved["rvq_combined_code_idx"]
                        np.testing.assert_array_equal(tokens[:, 0], np.repeat(np.arange(count), 6))
                        values = saved["body_pose"][:, 0]
                        self.assertEqual(len(values), count * 24)
                        np.testing.assert_array_equal(np.floor(values), np.repeat(np.arange(count), 24))
                        if not window:
                            poses, face = self.decode(self.torch.from_numpy(tokens))
                            np.testing.assert_array_equal(saved["body_pose"], poses["body_pose"])
                            np.testing.assert_array_equal(saved["motioncode"], face)

    def test_cancel_closes_token_iterator_and_does_not_save_partial_motion(self):
        """Cancellation must stop generation and leave no misleading final NPZ."""
        self.engine.decode_window_blocks = 8
        results = self.engine.generate(self.root, 25)
        next(results)
        next(results)
        results.close()
        self.assertEqual(len(self.generated), 8)
        self.assertEqual(self.closed, [True])
        self.assertFalse((self.root / "motion.npz").exists())


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class DemoRenderPipelineTest(unittest.TestCase):
    """Exercise lazy generation and continuous MP4/HLS encoding together."""

    def test_rasterization_overlaps_next_block_generation(self):
        """A rendering task must be active when the producer prepares its next block."""
        from demo.render import Renderer

        entered, release = threading.Event(), threading.Event()
        overlap = []

        def frames(vertices):
            """Hold rendering until the next motion block is being produced."""
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Rendering and generation did not overlap")
            return np.zeros((len(vertices), 720, 480, 3), dtype=np.uint8)

        def blocks():
            """Produce the second block while the first is still rasterizing."""
            yield np.zeros((24, 1, 3), dtype=np.float32)
            overlap.append(entered.wait(5) and not release.is_set())
            release.set()
            yield np.zeros((24, 1, 3), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(2) as pool:
            directory = Path(directory)
            sf.write(directory / "audio.wav", np.zeros((25600, 2)), 16000)
            renderer = Renderer.__new__(Renderer)
            renderer.pool = pool
            with patch("demo.render._frames", frames), (directory / "run.log").open("w") as log:
                segments = list(renderer.stream_vertices(blocks(), directory / "audio.wav", directory, log))
            self.assertEqual(overlap, [True])
            self.assertTrue(segments)
            self.assertTrue((directory / "video.mp4").is_file())
            self.assertIn("#EXT-X-ENDLIST", (directory / "segments/stream.m3u8").read_text())


    def test_segment_durations_match_video_and_partial_tail(self):
        """Avoiding ffprobe must preserve the HLS video clock, including the tail."""
        import json
        import subprocess
        from demo.app import segment_duration
        from demo.render import Renderer

        def frames(vertices):
            """Encode a known timeline without loading an OpenGL context."""
            return np.zeros((len(vertices), 720, 480, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(2) as pool:
            directory = Path(directory)
            sf.write(directory / "dialogue.wav", np.zeros((38400, 2)), 16000)
            renderer = Renderer.__new__(Renderer)
            renderer.pool = pool
            with patch("demo.render._frames", frames), (directory / "run.log").open("w") as log:
                segments = list(renderer.stream_vertices(iter([np.zeros((72, 1, 3))]),
                                                         directory / "dialogue.wav", directory, log))
            durations = [segment_duration(path) for path in segments]
            np.testing.assert_allclose(durations, [1.6, 0.8])
            for path, duration in zip(segments, durations):
                probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                                 "-show_entries", "stream=duration", "-of", "json", str(path)], text=True)
                self.assertAlmostEqual(duration, float(json.loads(probe)["streams"][0]["duration"]), places=4)


class DemoComparisonTest(unittest.TestCase):
    """Measure rotation differences correctly and reject mismatched token pairs."""

    def test_rotation_error_handles_axis_angle_wraparound(self):
        """179° and -179° about the same axis differ by 2°, rather than 358°."""
        from demo.comparison import compare_motion

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fields = {"body_pose": np.zeros((24, 63)), "left_hand_pose": np.zeros((24, 45)),
                      "right_hand_pose": np.zeros((24, 45)), "motioncode": np.zeros((24, 108)),
                      "rvq_combined_code_idx": np.zeros((6, 18), dtype=np.int64), "rvq_decode_window_blocks": 1}
            fields["body_pose"][:, 8] = np.deg2rad(179)
            np.savez(directory / "left.npz", **fields)
            fields["body_pose"][:, 8] = np.deg2rad(-179)
            np.savez(directory / "right.npz", **fields)
            metrics = compare_motion(directory / "left.npz", directory / "right.npz")
            self.assertAlmostEqual(metrics["upper_max_deg"], 2, places=5)
            self.assertAlmostEqual(metrics["upper_mean_deg"], 2 / 13, places=5)
            self.assertEqual(metrics["hands_mean_deg"], 0)
            fields["rvq_combined_code_idx"][0, 0] = 1
            np.savez(directory / "right.npz", **fields)
            with self.assertRaises(AssertionError):
                compare_motion(directory / "left.npz", directory / "right.npz")


class DemoSmoothingTest(unittest.TestCase):
    """Boundary smoothing must preserve rotations, history and untouched channels."""

    def test_rotation_wraparound_velocity_and_fade(self):
        """Blend on SO(3), retain angular velocity and reach the original block tail."""
        from scipy.spatial.transform import Rotation
        from demo.smoothing import smooth_boundary

        block = {key: np.zeros((24, width), dtype=np.float32)
                 for key, width in (("body_pose", 63), ("left_hand_pose", 45),
                                    ("right_hand_pose", 45), ("motioncode", 108))}
        previous = {key: value.copy() for key, value in block.items()}
        previous["body_pose"][-2:, 62] = np.deg2rad([179, -179])
        previous["left_hand_pose"][-2:, 2] = np.deg2rad([10, 12])
        block["body_pose"][:, 62] = np.deg2rad(-140)
        block["left_hand_pose"][:, 2] = np.deg2rad(60)
        original = {key: value.copy() for key, value in block.items()}
        history = {key: value.copy() for key, value in previous.items()}
        smooth_boundary(block, previous, 8)
        expected = Rotation.from_rotvec([0, 0, np.deg2rad(-177)])
        actual = Rotation.from_rotvec(block["body_pose"][0, 60:63])
        self.assertLess((expected.inv() * actual).magnitude(), 1e-6)
        self.assertAlmostEqual(float(np.rad2deg(block["left_hand_pose"][0, 2])), 14, places=4)
        for key in original:
            np.testing.assert_array_equal(block[key][7:], original[key][7:])
            np.testing.assert_array_equal(previous[key], history[key])
        np.testing.assert_array_equal(block["motioncode"], original["motioncode"])
        np.testing.assert_array_equal(block["body_pose"][:, :6], original["body_pose"][:, :6])

    def test_smoothed_preview_is_saved_exactly(self):
        """The renderer and NPZ must receive the same corrected motion arrays."""
        test = DemoMotionTest("test_zero_lookahead_commits_each_block_without_later_tokens")
        test.setUp()
        try:
            test.engine.decode_window_blocks = 1
            test.engine.boundary_smoothing_frames = 8
            list(test.engine.generate(test.root, 5))
            with np.load(test.root / "motion.npz") as saved:
                self.assertEqual(saved["motion_boundary_smoothing_frames"], 8)
                for key in test.rendered[0]:
                    np.testing.assert_array_equal(saved[key], np.concatenate([b[key] for b in test.rendered]))
        finally:
            test.doCleanups()


if __name__ == "__main__":
    unittest.main()
