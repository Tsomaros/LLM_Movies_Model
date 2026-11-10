import pytest
import torch

from llm_movies.model.loss import pinball_loss


def _batch():
    pred = torch.tensor(
        [
            [[0.0, 1.0, 3.0], [2.0, 3.0, 4.0]],
            [[1.0, 2.0, 2.5], [0.0, 1.0, 5.0]],
            [[-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]],
        ]
    )
    target = torch.tensor([[1.0, 2.5], [3.0, 0.5], [0.2, 4.0]])
    mask = torch.ones(3, 2, dtype=torch.bool)
    weight = torch.tensor([[1.0, 0.5], [1.0, 2.0], [1.0, 1.0]])
    return pred, target, mask, weight


def test_pinball_known_values():
    # target 1.0, preds (0, 1, 3): per-quantile 0.1, 0, 0.2 -> mean 0.1
    pred = torch.zeros(1, 2, 3)
    pred[0, 0] = torch.tensor([0.0, 1.0, 3.0])
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])
    weight = torch.ones(1, 2)
    assert pinball_loss(pred, target, mask, weight).item() == pytest.approx(0.1)


def test_loss_with_empty_target_mask():
    pred, target, mask, weight = _batch()
    mask[:, 1] = False
    only_rev = pinball_loss(pred, target, mask, weight)
    assert torch.isfinite(only_rev)
    mask[:, 0] = False
    assert pinball_loss(pred, target, mask, weight).item() == 0.0
    # a target with no rows contributes 0 and does not change the other term
    pred2 = pred.clone()
    pred2[:, 1] += 100
    mask[:, 0] = True
    assert pinball_loss(pred2, target, mask & torch.tensor([True, False]), weight).item() == pytest.approx(only_rev.item())


def test_score_weights_normalised():
    pred, target, mask, weight = _batch()
    base = pinball_loss(pred, target, mask, weight)
    scaled = pinball_loss(pred, target, mask, weight * 10)
    assert scaled.item() == pytest.approx(base.item())


def test_score_weights_matter_relatively():
    pred, target, mask, weight = _batch()
    other = weight.clone()
    other[1, 1] = 20.0
    assert pinball_loss(pred, target, mask, other).item() != pytest.approx(pinball_loss(pred, target, mask, weight).item())


def test_masked_rows_ignored():
    pred, target, mask, weight = _batch()
    mask[1, 0] = False
    mask[2, 1] = False
    base = pinball_loss(pred, target, mask, weight)
    target2 = target.clone()
    target2[1, 0] = 1e6
    target2[2, 1] = -1e6
    assert pinball_loss(pred, target2, mask, weight).item() == pytest.approx(base.item())
    # masked weights must not leak into normalisation either
    weight2 = weight.clone()
    weight2[2, 1] = 99.0
    assert pinball_loss(pred, target, mask, weight2).item() == pytest.approx(base.item())


def test_loss_is_differentiable():
    pred, target, mask, weight = _batch()
    pred.requires_grad_(True)
    pinball_loss(pred, target, mask, weight).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
