"""Check complete-window retention and fresh encoding across repeated context refreshes."""
import contextlib
import io
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from modeling_ellsa2 import CoSpeechGestureLM
from scripts import infer_ellsa2_gesture_expression_from_audio as inference


class Tokenizer:
    tokens = {'<|vision_start|>': 1, '<|vision_end|>': 2, '<|vision_pad|>': 3}

    def convert_tokens_to_ids(self, token):
        return self.tokens[token]

    def __call__(self, text, **kwargs):
        parts = re.split('(' + '|'.join(map(re.escape, self.tokens)) + ')', text)
        ids = torch.tensor([[self.tokens.get(part, 4) for part in parts if part]])
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


class WindowFbank:
    def extract_batch(self, audio, sampling_rate):
        # One feature per synthetic 0.8 s window, without changing input assembly.
        return [wav[::8].reshape(-1, 1) for wav in audio]


class Head(nn.Module):
    def forward(self, hidden):
        logits = hidden.new_zeros((*hidden.shape[:2], 256))
        return logits.scatter(-1, hidden[..., :1].long(), 1)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.chunk = 0
        self.calls = []

    def forward(self, inputs_embeds, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(last_hidden_state=torch.full_like(inputs_embeds, self.chunk))


class LongAudioTests(unittest.TestCase):
    def make_model(self):
        model = CoSpeechGestureLM.__new__(CoSpeechGestureLM)
        nn.Module.__init__(model)
        model.hidden_size = 2
        model.audio_tokens_per_window = 1
        model.motion_steps_per_window = 6
        model.num_motion_codebooks = 18
        model.motion_vocab_size = 256
        model.listener_fusion_delta = False
        model.text_embeddings = nn.Embedding(8, 2)
        model._embed_tokens = lambda ids: model.text_embeddings(ids)
        model.motion_embeds = nn.ModuleList(nn.Embedding(256, 2) for _ in range(18))
        model.motion_embed_weights = torch.ones(18)
        model.motion_heads = nn.ModuleList(Head() for _ in range(18))
        model.base_llm = nn.Module()
        model.base_llm.model = Backbone()
        model.audio_calls = []

        def encode_audio(features, lengths, listener=False):
            # Deliberately depend on all available left context. A stale slice
            # would differ from recomputing this sum from the retained crop.
            encoded = features.cumsum(dim=1).expand(-1, -1, 2)
            model.audio_calls.append((listener, features.clone(), encoded.clone()))
            if not listener:
                model.base_llm.model.chunk = int(features[0, -1, 0])
            return encoded, lengths

        model.encode_audio = encode_audio
        return model.eval()

    def run_stream(self, count, capacity=15, retained=3, dialog=True, style=True, live=False):
        args = inference.parse_args([])
        args.window_sec, args.audio_sr = 0.8, 10
        args.num_motion_codebooks, args.gesture_num_codebooks = 18, 12
        args.expression_steps_per_window = 6
        args.max_history_sec = capacity * args.window_sec
        args.reset_history_chunks = retained
        args.dialog = dialog
        model = self.make_model()
        model.dialog = dialog
        speaker = np.repeat(np.arange(1, count + 1, dtype=np.float32), 8)
        listener = speaker + 100 if dialog else np.zeros(0, dtype=np.float32)
        prefix = torch.full((6 if style else 0, 18), 255, dtype=torch.long)
        known = []
        original = inference.generate_motion_tokens_with_history

        def generate(**kwargs):
            known.append(kwargs['known_motion_ids'].clone())
            return original(**kwargs)

        with patch.object(inference, 'generate_motion_tokens_with_history', side_effect=generate), \
             contextlib.redirect_stdout(io.StringIO()):
            tokens, windows = inference.stream_generate_gesture_expression_tokens(
                speaker, listener, Tokenizer(), model, WindowFbank(), args,
                speaker_motion_ids=prefix,
                **({'audio_windows': zip(speaker.reshape(-1, 8), listener.reshape(-1, 8))} if live else {}))
        return tokens, windows, model, known, prefix

    def test_retains_three_completed_windows_and_reencodes_both_streams(self):
        for dialog in (True, False):
            for capacity, count in ((15, 3), (15, 15), (15, 16), (15, 40), (5, 10)):
                with self.subTest(dialog=dialog, capacity=capacity, count=count):
                    tokens, windows, model, known, prefix = self.run_stream(count, capacity, dialog=dialog)
                    self.assertEqual(windows, count)
                    self.assertEqual(tokens.shape, (count * 6, 18))
                    torch.testing.assert_close(tokens[:, 0], torch.arange(1, count + 1).repeat_interleave(6))
                    start = 0
                    for current in range(count):
                        if current - start + 1 > capacity:
                            start = current - 3
                        expected = torch.arange(start + 1, current + 1).repeat_interleave(6)
                        expected = expected[:, None].expand(-1, 18)
                        torch.testing.assert_close(known[current], torch.cat((prefix, expected)))
                        streams = 2 if dialog else 1
                        for stream in range(streams):
                            is_listener, features, encoded = model.audio_calls[current * streams + stream]
                            self.assertEqual(is_listener, bool(stream))
                            expected_audio = torch.arange(start + 1, current + 2).float() + stream * 100
                            torch.testing.assert_close(features[0, :, 0], expected_audio)
                            torch.testing.assert_close(encoded[0, :, 0], expected_audio.cumsum(0))
                        if current and start == current - 3:
                            self.assertEqual(known[current].size(0) - prefix.size(0), 18)
                    self.assertEqual(len(model.base_llm.model.calls), count * 6)
                    self.assertTrue(all(call['use_cache'] is False for call in model.base_llm.model.calls))

    def test_refresh_without_style_and_custom_retention(self):
        tokens, windows, model, known, prefix = self.run_stream(10, 5, retained=2, style=False)
        self.assertEqual(windows, 10)
        self.assertEqual(prefix.size(0), 0)
        torch.testing.assert_close(known[5][:, 0], torch.tensor([4] * 6 + [5] * 6))
        torch.testing.assert_close(known[8][:, 0], torch.tensor([7] * 6 + [8] * 6))
        torch.testing.assert_close(tokens[:, 0], torch.arange(1, 11).repeat_interleave(6))

    def test_live_iterator_matches_offline_without_requiring_a_final_duration(self):
        offline = self.run_stream(40)
        with patch.object(inference, 'compute_num_windows', side_effect=AssertionError('Live duration is unknown')):
            live = self.run_stream(40, live=True)
        torch.testing.assert_close(live[0], offline[0])
        for live_known, offline_known in zip(live[3], offline[3]):
            torch.testing.assert_close(live_known, offline_known)
        for live_call, offline_call in zip(live[2].audio_calls, offline[2].audio_calls):
            torch.testing.assert_close(live_call[1], offline_call[1])

    def test_retention_must_leave_room_for_current_window(self):
        for capacity in (2, 3):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(ValueError, 'leave room'):
                self.run_stream(4, capacity, retained=3)


if __name__ == '__main__':
    unittest.main()
