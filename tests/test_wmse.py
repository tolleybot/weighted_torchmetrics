# tests/functional/regression/test_wmse.py
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wmse import WeightedMeanSquaredError

# from torchmetrics.functional.regression.wmse import WeightedMeanSquaredError


def test_wmse():
    wmse = WeightedMeanSquaredError()
    preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
    target = torch.tensor([2.5, 5.0, 4.0, 8.0])
    weights = torch.tensor([1.0, 0.5, 2.0, 1.5])
    wmse.update(preds, target, weights)
    result = wmse.compute()
    expected = torch.tensor(1.25)  # replace with actual expected value
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"


if __name__ == "__main__":
    test_wmse()
    print("test_wmse passed")
