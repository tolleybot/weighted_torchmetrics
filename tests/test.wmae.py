import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wmae import WeightedMeanAbsoluteError


def test_wmae():
    # Initialize the metric
    wmae = WeightedMeanAbsoluteError()

    # Define sample predictions, targets, and weights
    preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
    targets = torch.tensor([2.5, 5.0, 4.0, 8.0])
    weights = torch.tensor([1.0, 0.5, 2.0, 1.5])

    # Update the metric with sample data
    wmae.update(preds, targets, weights)

    # Compute the result
    result = wmae.compute()

    # Define the expected result
    # Calculation:
    # Absolute errors: [0.5, 0.0, 1.5, 1.0]
    # Weighted absolute errors: [0.5*1.0, 0.0*0.5, 1.5*2.0, 1.0*1.5] = [0.5, 0.0, 3.0, 1.5]
    # Sum of weights: 1.0 + 0.5 + 2.0 + 1.5 = 5.0
    # Expected WMAE: (0.5 + 0.0 + 3.0 + 1.5) / 5.0 = 1.0
    expected = torch.tensor(1.0)

    # Assert that the computed result matches the expected result
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"


if __name__ == "__main__":
    test_wmae()
    print("test_wmae passed")
