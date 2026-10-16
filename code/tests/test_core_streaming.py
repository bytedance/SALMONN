"""Regression checks for audio expansion, shifted motion targets and gradient flow."""
import unittest

import torch
from torch import nn

from modeling_ellsa2 import CoSpeechGestureLM


class StreamingInputTests(unittest.TestCase):
    def setUp(self):
        """Build a small embedding-only model without loading external checkpoints."""
        self.model = CoSpeechGestureLM.__new__(CoSpeechGestureLM)
        nn.Module.__init__(self.model)
        self.model.hidden_size = 3
        self.model.audio_tokens_per_window = 2
        self.model.dialog = True
        self.model.listener_fusion_delta = False
        self.model.text_embeddings = nn.Embedding(8, 3)
        self.model._embed_tokens = lambda ids: self.model.text_embeddings(ids)
        self.model.encode_audio = lambda features, lengths, listener=False: (features, lengths)
        self.speaker = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3).requires_grad_()
        self.listener = (self.speaker.detach() + 20).requires_grad_()
        self.motion = torch.randn(2, 3, 3, requires_grad=True)
        ids = torch.tensor([[1, 2, 3, 4, 5, 5, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]])
        starts = torch.zeros_like(ids, dtype=torch.bool)
        ends = torch.zeros_like(starts)
        slots = torch.zeros_like(starts)
        starts[:, 1] = True
        ends[:, 2] = True
        starts[0, 6] = ends[0, 7] = True
        slots[0, [4, 5, 9]] = True
        slots[1, 4] = True
        self.slots = slots
        self.inputs = dict(
            input_ids=ids, attention_mask=(ids != 0).long(),
            audio_start_mask=starts, audio_end_mask=ends,
            fbank_feature=self.speaker, fbank_feature_len=torch.tensor([3, 2]),
            audio_listen_fbank=self.listener, audio_listen_fbank_len=torch.tensor([1, 1]),
        )

    def test_paired_streams_padding_and_shifted_targets(self):
        """Alternate speaker/listener chunks and shift motion targets by one position."""
        embeds, attention, targets = self.model.prepare_inputs_labels_for_speech(
            **self.inputs, motion_slot_mask=self.slots, motion_input_embeds=self.motion,
        )
        self.assertEqual(embeds.shape, (2, 13, 3))
        self.assertEqual(attention.sum(dim=1).tolist(), [13, 7])
        self.assertEqual(targets[0].nonzero().flatten().tolist(), [5, 6, 11])
        self.assertEqual(targets[1].nonzero().flatten().tolist(), [5])
        torch.testing.assert_close(embeds[0, 2:4], self.speaker[0, :2])
        torch.testing.assert_close(embeds[0, 9:10], self.listener[0, :1])
        torch.testing.assert_close(embeds[0, [6, 7, 12]], self.motion[0])
        self.assertFalse(embeds[1, 7:].any())

    def test_joint_labels_and_gradients(self):
        """Ignore inserted audio labels and keep gradients for every used embedding."""
        labels = self.inputs['input_ids'].to(torch.int32)
        embeds, _, _, expanded = self.model.prepare_inputs_labels_for_speech_and_motion(
            **self.inputs, motion_slot_mask=self.slots, motion_input_embeds=self.motion, labels=labels,
        )
        self.assertEqual(expanded.dtype, torch.long)
        self.assertEqual(expanded[0].tolist(), [1, 2, -100, -100, 3, 4, 5, 5, 2, -100, 3, 4, 5])
        self.assertTrue((expanded[1, 7:] == -100).all())
        embeds.sum().backward()
        torch.testing.assert_close(self.speaker.grad.sum(dim=-1), torch.tensor([[3., 3., 0.], [3., 3., 0.]]))
        torch.testing.assert_close(self.listener.grad.sum(dim=-1), torch.tensor([[3., 0., 0.], [0., 0., 0.]]))
        torch.testing.assert_close(self.motion.grad.sum(dim=-1), torch.tensor([[3., 3., 3.], [3., 0., 0.]]))
        self.assertIsNotNone(self.model.text_embeddings.weight.grad)

    def test_single_stream_short_final_chunk(self):
        """Use the final partial speaker chunk and zero unknown motion inputs."""
        self.model.dialog = False
        embeds, attention, targets = self.model.prepare_inputs_labels_for_speech(
            **self.inputs, motion_slot_mask=self.slots,
        )
        self.assertEqual(attention.sum(dim=1).tolist(), [13, 7])
        torch.testing.assert_close(embeds[0, 9:10], self.speaker[0, 2:3])
        self.assertFalse(embeds[targets.roll(1, dims=1)].any())

    def test_exhausted_audio_fails(self):
        """Raise at an audio placeholder when its source stream has no frames."""
        self.inputs['fbank_feature_len'] = torch.tensor([0, 2])
        with self.assertRaisesRegex(ValueError, 'Not enough audio embeddings'):
            self.model.prepare_inputs_labels_for_speech(**self.inputs, motion_slot_mask=self.slots)


if __name__ == '__main__':
    unittest.main()
