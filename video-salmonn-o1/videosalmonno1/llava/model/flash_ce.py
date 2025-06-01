import logging

import torch

try:
    from bumi.function import flash_cross_entropy
except ImportError:
    flash_cross_entropy = None
    logging.warning("Bumi not installed correctly, performance may suffer.")


def native_cross_entropy(x, w, labels):
    logits = torch.nn.functional.linear(x, w, None)
    loss = torch.nn.functional.cross_entropy(logits.to(torch.float32), labels, reduction="none")
    with torch.no_grad():
        acc = (logits.max(dim=1)[1] == labels).float()
    return loss, acc


def flash_cross_entropy_func(x, w, labels, use_flash_ce=True):
    # assert x.dim() == labels.dim() + 1, f"{x.shape}, {labels.shape}"
    x = x.squeeze(0)
    if x.dim() == 3:
        bsz = x.shape[0]
        x = x.reshape(-1, x.shape[-1])
        labels = labels.reshape(-1)
    else:
        bsz = None

    def reshape_if_needed(loss, acc):
        if bsz is None:
            return loss, acc
        return loss.reshape(bsz, -1), acc.reshape(bsz, -1)

    if flash_cross_entropy is not None and use_flash_ce:
        # set debug=True to align the logits precision with native cross entropy
        return reshape_if_needed(
            *flash_cross_entropy(x.to(w.dtype), w, labels, recompute_level=2, calc_acc=True, debug=True)
        )
    else:
        return reshape_if_needed(*native_cross_entropy(x, w, labels))
