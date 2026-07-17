"""Self checks for repro.py. Run: python test_repro.py"""

import math
import tempfile
from pathlib import Path

import torch

from repro import assert_no_overlap, merge_taps, reweighted_score


def test_reweighting_toy():
    # q* has squared distances 1, 2, 3 to the three bank neighbours of n*,
    # with d(q*, n*) = 1. By hand: w = 1 - e^1 / (e^1 + e^2 + e^3).
    expected = (1.0 - math.exp(1) / (math.exp(1) + math.exp(2) + math.exp(3))) * 1.0
    got = reweighted_score(1.0, [1.0, 2.0, 3.0])
    assert abs(got - expected) < 1e-12, (got, expected)


def test_reweighting_uniform():
    # all b neighbours equidistant: w = 1 - 1/b
    got = reweighted_score(5.0, [5.0, 5.0, 5.0])
    assert abs(got - (1.0 - 1.0 / 3.0) * 5.0) < 1e-12, got


def test_reweighting_stable():
    # squared distances in the tens of thousands must not overflow
    got = reweighted_score(4e4, [4e4, 4.5e4, 5e4])
    assert math.isfinite(got), got
    assert 0.0 <= got <= 4e4, got


def test_reweighting_needs_l2_not_squared():
    # Regression guard for a real bug. Eq. 7 weights the L2 norm. Feed it
    # squared distances and exp() saturates, w pins at 1.0, and the image
    # score quietly becomes plain max-min-distance. The reweighting still
    # runs and still passes the toy tests above, it just does nothing.
    d = [5.0, 5.4, 5.9, 6.3, 6.8, 7.2, 7.7, 8.1, 8.6]
    w_l2 = reweighted_score(d[0], d) / d[0]
    sq = [x * x for x in d]
    w_sq = reweighted_score(sq[0], sq) / sq[0]
    assert w_l2 < 0.995, f"L2 reweighting should bite, got w={w_l2}"
    assert w_sq > 0.9999, f"squared input should saturate, got w={w_sq}"


def test_patch_grid_shape():
    # layer2/layer3 shapes for a 224 input, must give 784 patches x 1536 dims
    f2 = torch.randn(2, 512, 28, 28)
    f3 = torch.randn(2, 1024, 14, 14)
    feats, grid = merge_taps(f2, f3)
    assert grid == (28, 28), grid
    assert feats.shape == (2, 784, 1536), feats.shape


def test_hash_overlap_detected():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a = tmp / "train_000.png"
        b = tmp / "test_000.png"
        c = tmp / "test_clean.png"
        a.write_bytes(b"same bytes")
        b.write_bytes(b"same bytes")
        c.write_bytes(b"different bytes")
        assert_no_overlap([a], [c])
        try:
            assert_no_overlap([a], [b])
        except AssertionError:
            pass
        else:
            raise AssertionError("byte-identical file across splits was not caught")


if __name__ == "__main__":
    test_reweighting_toy()
    test_reweighting_uniform()
    test_reweighting_stable()
    test_reweighting_needs_l2_not_squared()
    test_patch_grid_shape()
    test_hash_overlap_detected()
    print("all checks passed")
