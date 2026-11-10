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


def test_pinball_known_values_weighted_both_targets():
    # preds (0, 1, 3) for every cell; pinball(t) = mean over q of max(q*(t-p), (q-1)*(t-p))
    # t=1: 0.1, 0, 0.2 -> 0.1 ; t=3: 0.3, 1.0, 0 -> 13/30
    pred = torch.tensor([[0.0, 1.0, 3.0]]).expand(2, 2, 3).clone()
    target = torch.tensor([[1.0, 3.0], [3.0, 1.0]])
    mask = torch.tensor([[True, True], [True, False]])
    weight = torch.tensor([[1.0, 5.0], [3.0, 7.0]])
    # revenue: rows (t=1, w=1) and (t=3, w=3): (1*0.1 + 3*13/30) / 4 = 0.35
    # score: only row 0 (t=3): 13/30 (its weight normalises to 1)
    expected = 0.35 + 13 / 30
    assert pinball_loss(pred, target, mask, weight).item() == pytest.approx(expected, abs=1e-6)


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


def test_empty_mask_gives_finite_zero_gradients():
    pred, target, mask, weight = _batch()
    pred.requires_grad_(True)
    # one target empty: its gradient is exactly 0, the other target still learns
    one_empty = mask.clone()
    one_empty[:, 1] = False
    loss = pinball_loss(pred, target, one_empty, weight)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad[:, 1] == 0).all()
    assert (pred.grad[:, 0] != 0).any()
    # both empty (also with zero weights): loss exactly 0, gradients finite and 0
    pred.grad = None
    none = torch.zeros_like(mask)
    loss = pinball_loss(pred, target, none, torch.zeros_like(weight))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad == 0).all()


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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_masked_nan_inf_target_and_weight_ignored(bad):
    pred, target, mask, weight = _batch()
    mask[1, 0] = False
    mask[2, 1] = False
    pred.requires_grad_(True)
    base = pinball_loss(pred, target, mask, weight)
    base.backward()
    base_grad = pred.grad.clone()
    pred.grad = None
    target2, weight2 = target.clone(), weight.clone()
    target2[1, 0] = bad
    target2[2, 1] = bad
    loss = pinball_loss(pred, target2, mask, weight2)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(base.item())
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.allclose(pred.grad, base_grad)
    assert (pred.grad[1, 0] == 0).all() and (pred.grad[2, 1] == 0).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_masked_nan_inf_weight_ignored(bad):
    # a film with no score may carry a NaN weight: it must not poison the loss
    pred, target, mask, weight = _batch()
    mask[2, 1] = False
    base = pinball_loss(pred, target, mask, weight)
    weight2 = weight.clone()
    weight2[2, 1] = bad
    loss = pinball_loss(pred, target, mask, weight2)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(base.item())


def test_loss_is_differentiable():
    pred, target, mask, weight = _batch()
    pred.requires_grad_(True)
    pinball_loss(pred, target, mask, weight).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
