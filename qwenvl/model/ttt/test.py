import torch
from tqdm import tqdm
from qwenvl.model.ttt.configs import ModelConfig
from qwenvl.model.ttt.ttt_layer import TTTWrapper
from qwenvl.model.ttt.cogvideo_utils import SequenceMetadata
import time


def test_ttt(ttt, niter=100, hdim=512):
    start = time.time()
    for i in tqdm(range(niter)):
        # video_inputs = torch.rand(1, 10, 3, hdim, hdim).cuda()
        all_inputs = torch.rand(1, 16384, hdim).cuda().bfloat16()

        seq_metadata = SequenceMetadata(
            text_length=0,
            seq_text_length=0 * 1,
            num_frames=0,
            num_chunks=1,
            tokens_per_frame=1,
        )
        output = ttt(all_inputs, seq_metadata)
        loss = output.mean()
        loss.backward()
    print("Time elapsed: {:.2f}".format(time.time()-start))


if __name__ == "__main__":
    hdim = 3072
    configs = ModelConfig(model_dim=hdim, num_heads=3072//64, num_layers=1)
    # ttt = TTTWrapper(configs, use_kernel=True).cuda().bfloat16()
    # print("Test with kernel")
    # test_ttt(ttt, niter=1000, hdim=hdim)
    ttt = TTTWrapper(configs, use_kernel=False).cuda().bfloat16()
    print("Test without kernel")
    test_ttt(ttt, niter=1000, hdim=hdim)