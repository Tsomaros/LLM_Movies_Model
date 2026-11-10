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
    out = head(torch.randn(1000, 16) * 10)
    assert (out[..., 0] <= out[..., 1]).all()
    assert (out[..., 1] <= out[..., 2]).all()
