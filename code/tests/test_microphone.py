"""Real microphone buffering, resampling and continuous video encoding checks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np
import soxr

from demo.microphone import LiveAudio


class MicrophoneAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = LiveAudio(self.root / 'live', 60)
        self.addCleanup(self.audio.cancel)

    def test_complete_windows_are_available_before_recording_stops(self):
        values = np.arange(32000, dtype=np.int16)
        self.audio.feed((16000, values[:6400]))
        self.assertTrue(self.audio.queue.empty())
        self.audio.feed((16000, values[6400:12800]))
        block, partner = self.audio.queue.get_nowait()
        self.assertFalse(self.audio.finished)
        np.testing.assert_array_equal(block, values[:12800].astype(np.float32) / 32768)
        self.assertFalse(partner.any())
        self.audio.feed((16000, values[12800:]))
        self.audio.finish()
        tail = list(self.audio)
        self.assertEqual(len(tail), 2)
        self.assertFalse(tail[-1][0][6400:].any())
        with wave.open(str(self.audio.directory / 'dialogue.wav')) as handle:
            self.assertEqual((handle.getnframes(), handle.getnchannels()), (38400, 2))
            saved = np.frombuffer(handle.readframes(38400), '<i2').reshape(-1, 2)
        np.testing.assert_array_equal(saved[:32000, 0], values)
        self.assertFalse(saved[:, 1].any())

    def test_stream_resampling_matches_complete_clip_without_sample_loss(self):
        rate = 48000
        samples = (np.sin(np.arange(100800) * 0.1) * 10000).astype(np.int16)
        for start in range(0, len(samples), 7919):
            stereo = np.column_stack((samples[start:start + 7919], samples[start:start + 7919]))
            self.audio.feed((rate, stereo))
        self.audio.finish()
        actual = np.concatenate([target for target, _ in self.audio])
        expected = soxr.resample(samples.astype(np.float32) / 32768, rate, 16000)
        self.assertEqual(len(actual), 38400)
        np.testing.assert_allclose(actual[:len(expected)], expected, atol=1e-6, rtol=1e-6)
        self.assertFalse(actual[len(expected):].any())

    def test_stop_recovers_unsent_tail_once(self):
        values = np.full(30000, 7000, dtype=np.int16)
        self.audio.feed((16000, values[:22000]))
        self.audio.finish((16000, values))
        self.audio.finish((16000, values))
        self.audio.feed((16000, values))
        blocks = list(self.audio)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(self.audio.source_frames, 30000)
        with wave.open(str(self.audio.directory / 'target.wav')) as handle:
            self.assertEqual(handle.getnframes(), 38400)

    def test_duration_cap_and_cancel_release_the_consumer(self):
        self.audio.max_seconds = 1.0
        self.audio.feed((16000, np.zeros(24000, dtype=np.int16)))
        self.assertTrue(self.audio.finished)
        self.assertEqual(self.audio.source_frames, 16000)
        self.assertEqual(len(list(self.audio)), 2)
        other = LiveAudio(self.root / 'cancel', 60)
        other.cancel()
        self.assertEqual(list(other), [])


class MicrophoneTransportTests(unittest.TestCase):
    def test_ordered_packets_and_partial_tail_reach_recording_exactly_once(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from demo.microphone import install_microphone_routes

        with tempfile.TemporaryDirectory() as directory:
            audio = LiveAudio(Path(directory) / 'socket', 60)
            app = FastAPI()
            install_microphone_routes(app, lambda key: audio if key == 'session' else None)
            values = np.sin(np.arange(100813, dtype=np.float32) * 0.1) * 0.25
            with TestClient(app) as client:
                self.assertEqual(client.get('/microphone/processor.js').status_code, 200)
                with client.websocket_connect('/microphone/socket/session') as socket:
                    socket.send_json({'sample_rate': 48000})
                    for start in range(0, len(values), 19200):
                        socket.send_bytes(values[start:start + 19200].astype('<f4').tobytes())
                    socket.send_text('stop')
                    self.assertEqual(socket.receive_json(), {'type': 'finished', 'source_frames': len(values)})
            actual = np.concatenate([block for block, _ in audio])
            expected = soxr.resample(values, 48000, 16000)
            np.testing.assert_allclose(actual[:len(expected)], expected, atol=1e-6, rtol=1e-6)
            self.assertFalse(actual[len(expected):].any())

    def test_disconnect_flushes_received_audio(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from demo.microphone import install_microphone_routes

        with tempfile.TemporaryDirectory() as directory:
            audio = LiveAudio(Path(directory) / 'socket', 60)
            app = FastAPI()
            install_microphone_routes(app, lambda key: audio)
            with TestClient(app) as client:
                with client.websocket_connect('/microphone/socket/session') as socket:
                    socket.send_json({'sample_rate': 16000})
                    socket.send_bytes(np.full(17001, 0.25, dtype='<f4').tobytes())
            self.assertTrue(audio.finished)
            self.assertEqual(audio.source_frames, 17001)
            blocks = list(audio)
            self.assertEqual(len(blocks), 2)
            self.assertTrue((blocks[1][0][:4201] == 0.25).all())
            self.assertFalse(blocks[1][0][4201:].any())


class MotionPrefixTests(unittest.TestCase):
    def test_dialogue_prefix_references_include_both_roles(self):
        from demo.prefixes import load_motion_prefixes, DEFAULT_PREFIX
        from release_paths import DATA_ROOT

        prefixes = load_motion_prefixes()
        self.assertIn(DEFAULT_PREFIX, prefixes)
        selected = json.loads((DATA_ROOT / 'manifests/selected_speakers.json').read_text())
        for key, prefix in prefixes.items():
            self.assertEqual(prefix['in_training_inventory'], prefix['speaker_id'] in selected)
            self.assertGreater(prefix['speaking_seconds'], 3)
            self.assertGreater(prefix['listening_seconds'], 3)
            with np.load(prefix['pose_path']) as pose:
                self.assertEqual(len(pose['smplh:body_pose']), prefix['source_frames'])
                self.assertTrue(pose['smplh:is_valid'].all())

    def test_switching_live_prefix_does_not_change_other_requests_or_upload_style(self):
        import torch
        from types import SimpleNamespace
        from demo.engine import Engine, inference
        from demo.prefixes import load_motion_prefixes

        engine = Engine.__new__(Engine)
        engine.args = SimpleNamespace()
        engine.decoders, engine.flame = {}, None
        engine.tokenizer = engine.model = engine.fbank = None
        engine.style = torch.full((6, 18), 77, dtype=torch.long)
        engine.prefixes = load_motion_prefixes()
        engine.prefix_tokens = {key: torch.full((6, 18), index, dtype=torch.long)
                                for index, key in enumerate(engine.prefixes)}
        seen = []

        def token_stream(*unused, speaker_motion_ids, audio_windows, **kwargs):
            seen.append(speaker_motion_ids.clone())
            for target, partner in audio_windows:
                self.assertFalse(partner.any())
                yield speaker_motion_ids

        def render(directory, tokens, args, windows, audio_path):
            self.assertIsNone(windows)
            self.assertEqual(args.motion_style_start, 0)
            list(tokens)
            yield None, [str(directory / 'motion.npz'), str(directory / 'video.mp4')], 'Done'

        engine._render_tokens = render
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(inference, 'use_streaming_rvq', return_value=True), \
                patch.object(inference, 'iter_gesture_expression_tokens', side_effect=token_stream):
            keys = list(engine.prefixes)
            for key in [*keys, keys[0]]:
                audio = LiveAudio(Path(directory) / str(len(seen)), 60)
                audio.feed((16000, np.zeros(12800, np.float32)))
                audio.finish()
                result = list(engine.generate_microphone(audio, key))[-1]
                self.assertTrue(torch.equal(seen[-1], engine.prefix_tokens[key]))
                metadata = json.loads(Path(result[1][-1]).read_text())
                self.assertEqual(metadata['id'], key)
                self.assertEqual(metadata['reference_pose'], engine.prefixes[key]['reference_pose'])
                self.assertFalse('pose_path' in metadata)
        self.assertTrue(torch.equal(engine.style, torch.full((6, 18), 77, dtype=torch.long)))
        self.assertTrue(torch.equal(seen[0], seen[-1]))
        self.assertTrue(all(not torch.equal(seen[0], value) for value in seen[1:-1]))


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required')
class LiveRendererTests(unittest.TestCase):
    def test_muted_segments_are_published_before_stream_finishes(self):
        from demo.render import Renderer
        generated = []

        def blocks():
            for index in range(8):
                generated.append(index)
                yield np.zeros((24, 1, 3), dtype=np.float32)

        def frames(vertices):
            return np.zeros((len(vertices), 720, 480, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(2) as pool:
            directory = Path(directory)
            renderer = Renderer.__new__(Renderer)
            renderer.pool = pool
            with patch('demo.render._frames', frames), (directory / 'run.log').open('w') as log:
                stream = renderer.stream_vertices(blocks(), None, directory, log, segment_frames=24, prefetch_batches=1)
                first = next(stream)
                self.assertLess(len(generated), 8)
                segments = [first, *stream]
            self.assertEqual(len(segments), 8)
            self.assertIn('#EXT-X-ENDLIST', (directory / 'segments/stream.m3u8').read_text())
            probe = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(directory / 'video.mp4')]))
            self.assertEqual(len(probe['streams']), 1)
            self.assertEqual(int(probe['streams'][0]['nb_read_frames']), 192)
            self.assertEqual(float(probe['streams'][0]['duration']), 6.4)


if __name__ == '__main__':
    unittest.main()
