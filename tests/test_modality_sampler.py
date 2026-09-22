"""Regression tests for qwenvl.data.modality_sampler.WeightedRoundRobinBatchSampler.

Only needs `torch`; runnable either with pytest or directly:
    python tests/test_modality_sampler.py
"""

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwenvl.data.modality_sampler import WeightedRoundRobinBatchSampler


class _TypeListDataset:
    """Minimal stand-in for LazySupervisedDataset: the sampler only reads .type_list / len()."""

    def __init__(self, type_list):
        self.type_list = type_list

    def __len__(self):
        return len(self.type_list)


def _build(type_list, batch_size):
    """Return (sampler, captured_stdout)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        sampler = WeightedRoundRobinBatchSampler(_TypeListDataset(type_list), batch_size, seed=0)
    return sampler, buf.getvalue()


def test_modality_smaller_than_batch_size_is_reported():
    # 40 audio-only samples but a global batch of 128 -> the whole bucket is skipped.
    type_list = ["t"] * 5000 + ["v"] * 5000 + ["a"] * 40
    sampler, out = _build(type_list, 128)

    emitted = list(iter(sampler))
    assert not any(type_list[i] == "a" for i in emitted), "'a' samples unexpectedly emitted"
    # Dropping them silently is the bug: the user must be told.
    assert "WARNING" in out and "'a'" in out, f"no warning emitted, got:\n{out}"


def test_empty_dataset_raises_value_error():
    try:
        _build([], 128)
    except ValueError as exc:
        assert "empty dataset" in str(exc)
    except ZeroDivisionError:
        raise AssertionError("empty dataset raised a bare ZeroDivisionError")
    else:
        raise AssertionError("empty dataset did not raise")


def test_balanced_dataset_behaviour_is_unchanged():
    type_list = ["av"] * 2000 + ["v"] * 2000 + ["a"] * 2000 + ["t"] * 2000
    sampler, out = _build(type_list, 64)

    emitted = list(iter(sampler))
    assert len(emitted) == len(sampler) == 4 * (2000 // 64) * 64
    assert len(set(emitted)) == len(emitted), "an index was sampled twice"
    for modality in ("av", "v", "a", "t"):
        assert sum(type_list[i] == modality for i in emitted) == (2000 // 64) * 64
    assert "WARNING" not in out, f"unexpected warning:\n{out}"
    assert all(len(batch) == 64 for batch in sampler.out_data)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
