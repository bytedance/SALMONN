"""Exercise bounded causal RVQ state across block and context boundaries."""
from types import SimpleNamespace
import unittest

import torch

from models.vq.model import RVQVAE
from models.vq.streaming import RVQDecoderStream


class RVQStreamingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        args = SimpleNamespace(causal=True, num_quantizers=2, shared_codebook=False,
                               quantize_dropout_prob=0, mu=0.99)
        self.model = RVQVAE(args, input_width=3, nb_code=16, code_dim=8,
                           output_emb_width=8, down_t=2, stride_t=2,
                           width=8, depth=2, dilation_growth_rate=2).eval().set_context_frames(12)
        for layer in self.model.quantizer.layers:
            layer.codebook.normal_()
        self.codes = torch.randint(0, 16, (2, 21, 2))

    def test_stream_matches_full_decoding_across_context_boundaries(self):
        with torch.no_grad():
            expected = self.model.forward_decoder(self.codes)
            for sizes in ([1] * 21, [6, 6, 6, 3], [2, 5, 1, 7, 6]):
                stream, outputs, left = RVQDecoderStream(self.model), [], 0
                for size in sizes:
                    right = left + size
                    block = stream.forward_decoder(self.codes[:, left:right])
                    self.assertEqual(block.shape[1], size * 4)
                    prefix = self.model.forward_decoder(self.codes[:, :right])[:, left * 4:]
                    torch.testing.assert_close(block, prefix, atol=1e-6, rtol=1e-6)
                    self.assertLessEqual(stream.cached_steps, 2)
                    self.assertEqual(stream.history.untyped_storage().nbytes(), stream.history.numel() * stream.history.element_size())
                    outputs.append(block)
                    left = right
                torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-6, rtol=1e-6)

    def test_requests_have_independent_states_and_no_future_input(self):
        left, right = RVQDecoderStream(self.model), RVQDecoderStream(self.model)
        first = left.forward_decoder(self.codes[:, :6]).clone()
        torch.testing.assert_close(right.forward_decoder(self.codes[:, :6]), first)
        left.forward_decoder(self.codes[:, 6:12])
        self.assertEqual(right.cached_steps, 2)
        torch.testing.assert_close(right.forward_decoder(self.codes[:, 6:12]),
                                   self.model.forward_decoder(self.codes[:, :12])[:, 24:], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(first, self.model.forward_decoder(self.codes)[:, :24], atol=1e-6, rtol=1e-6)

    def test_one_step_cap_keeps_no_history(self):
        self.model.set_context_frames(4)
        stream = RVQDecoderStream(self.model)
        outputs = [stream.forward_decoder(block) for block in self.codes.split(6, dim=1)]
        self.assertEqual(stream.cached_steps, 0)
        torch.testing.assert_close(torch.cat(outputs, dim=1), self.model.forward_decoder(self.codes), atol=1e-6, rtol=1e-6)

    def test_noncausal_or_uncapped_models_are_rejected(self):
        self.model.causal = False
        with self.assertRaises(ValueError):
            RVQDecoderStream(self.model)
        self.model.causal = True
        self.model.set_context_frames(0)
        with self.assertRaises(ValueError):
            RVQDecoderStream(self.model)


if __name__ == "__main__":
    unittest.main()
