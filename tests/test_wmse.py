# tests/functional/regression/test_wmse.py
import torch
from torchmetrics.functional.regression.wmse import WeightedMeanSquaredError


def test_wmse():
    wmse = WeightedMeanSquaredError()
    preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
    target = torch.tensor([2.5, 5.0, 4.0, 8.0])
    weights = torch.tensor([1.0, 0.5, 2.0, 1.5])
    wmse.update(preds, target, weights)
    result = wmse.compute()
    expected = torch.tensor(0.8750)  # replace with actual expected value
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"
