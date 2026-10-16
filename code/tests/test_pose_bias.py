"""Reference validation prevents timing shifts and fabricated face guidance."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.pose_bias import pose_guide_schedule


class PoseBiasTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "pose.npz"
        self.pose = {name: np.zeros((200, count, 3), dtype=np.float32)
                     for name, count in (("body_pose", 21), ("left_hand_pose", 15), ("right_hand_pose", 15))}
        np.savez(self.path, **self.pose)

    def test_pose_alone_needs_no_expression_or_translation(self):
        schedule = pose_guide_schedule(self.path, 8, 3.2, 25)
        reference = schedule[4]["reference"]
        self.assertEqual(reference["causal_reference_start_frame"], 8)
        self.assertEqual(reference["causal_reference_length_frames"], 192)
        self.assertNotIn("flame_file", reference)

    def test_namespaced_pose_keys(self):
        np.savez(self.path, **{f"smplh:{key}": value for key, value in self.pose.items()})
        self.assertEqual(list(pose_guide_schedule(self.path, 0, 0.0, 8)), [0])

    def test_invalid_timing_is_not_rounded_or_truncated(self):
        for time in (-0.8, 0.3, float("nan"), float("inf"), 14.4):
            with self.subTest(time=time), self.assertRaises(ValueError):
                pose_guide_schedule(self.path, 0, time, 25)
        for start in (-1, 0.5, 9):
            with self.subTest(start=start), self.assertRaises(ValueError):
                pose_guide_schedule(self.path, start, 3.2, 25)

    def test_encoder_masks_face_without_loading_face_or_translation(self):
        from types import SimpleNamespace
        import torch
        from scripts import infer_ellsa2_gesture_expression_from_audio as inference

        class Encoder(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.value = value

            def encode(self, features):
                self.seen = features.shape
                return torch.full((1, features.shape[1] // 4, 6), self.value), None

        # Exported motion is flattened; it must also work as a new guide.
        np.savez(self.path, **{key: value.reshape(len(value), -1) for key, value in self.pose.items()})
        pose_guide_schedule(self.path, 0, 3.2, 25)
        # The existing rotation/statistics path is real; only RVQ weights are replaced.
        decoders = {"upper": Encoder(3), "hands": Encoder(7)}
        stats = inference.load_gesture_stats(inference.DEFAULT_STATS_DIR,
                                             inference.DEFAULT_STATS_PREFIX, inference.DEFAULT_STATS_SUFFIX)
        args = SimpleNamespace(interleave_motion_parts=False, motion_tokenizer_type="upper_hands",
                               gesture_num_codebooks=12, num_motion_codebooks=18)
        codes = inference.encode_pose_tokens(str(self.path), 0, 192, decoders, stats, args)
        self.assertEqual(tuple(codes.shape), (48, 18))
        self.assertTrue(torch.all(codes[:, :6] == 3))
        self.assertTrue(torch.all(codes[:, 6:12] == 7))
        self.assertTrue(torch.all(codes[:, 12:] == -100))
        self.assertEqual(tuple(decoders["upper"].seen), (1, 192, 78))
        self.assertEqual(tuple(decoders["hands"].seen), (1, 192, 180))

    def test_model_selection_preserves_default_and_explicit_checkpoint(self):
        from scripts import infer_ellsa2_gesture_expression_from_audio as inference
        base = inference.finalize_args(inference.parse_args([]))
        self.assertEqual(base.model_path, inference.DEFAULT_MODEL_PATH)
        for options in (["--pose-guide", str(self.path)], ["--motion-bias"]):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "--model-path"):
                inference.finalize_args(inference.parse_args(options))
        explicit = inference.finalize_args(inference.parse_args([
            "--pose-guide", str(self.path), "--model-path", "custom-checkpoint"]))
        self.assertEqual(explicit.model_path, "custom-checkpoint")
        self.assertTrue(explicit.motion_bias and explicit.inline_exemplar_enabled)
        with self.assertRaisesRegex(ValueError, "learned"):
            inference.finalize_args(inference.parse_args([
                "--pose-guide", str(self.path), "--inline-exemplar-copy-mode", "hard_core"]))

    def test_bad_pose_and_frame_rate_fail(self):
        self.pose["left_hand_pose"][100, 0, 0] = np.nan
        np.savez(self.path, **self.pose)
        with self.assertRaises(ValueError):
            pose_guide_schedule(self.path, 0, 3.2, 25)
        np.savez(self.path, **self.pose, mocap_frame_rate=60)
        with self.assertRaisesRegex(ValueError, "30 fps"):
            pose_guide_schedule(self.path, 0, 3.2, 25)


if __name__ == "__main__":
    unittest.main()
