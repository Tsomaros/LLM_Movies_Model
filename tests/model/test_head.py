import torch

from llm_movies.model import QUANTILES, TARGETS
from llm_movies.model.head import QuantileHead


def test_constants():
    assert QUANTILES == (0.1, 0.5, 0.9)
    assert TARGETS == ("revenue", "score")


def test_head_shape():
    out = QuantileHead(16, hidden=8)(torch.randn(5, 16))
    assert out.shape == (5, 2, 3)


def test_quantiles_never_cross():
    torch.manual_seed(0)
    head = QuantileHead(16, hidden=8)
    # scales from ordinary up to extreme, several seeds, plus all-zero and constant inputs
    inputs = [torch.randn(1000, 16) * s for s in (1.0, 10.0, 1e3, 1e6)]
    inputs += [torch.zeros(10, 16), torch.full((10, 16), -1e6), torch.full((10, 16), 1e6)]
    for seed in range(3):
        torch.manual_seed(seed)
        head = QuantileHead(16, hidden=8)
        for x in inputs:
            out = head(x)
            assert torch.isfinite(out).all()
            assert (out[..., 0] <= out[..., 1]).all()
            assert (out[..., 1] <= out[..., 2]).all()


def test_head_is_not_degenerate():
    # the spread is learned, not collapsed: for ordinary inputs the bounds sit strictly around the median
    torch.manual_seed(0)
    out = QuantileHead(16, hidden=8)(torch.randn(100, 16))
    assert (out[..., 0] < out[..., 1]).all()
    assert (out[..., 1] < out[..., 2]).all()
