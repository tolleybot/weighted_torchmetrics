# torchmetrics/functional/regression/wmae.py
import torch
from torchmetrics import Metric


class WeightedMeanAbsoluteError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "sum_absolute_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("sum_weights", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        absolute_error = torch.abs(preds - target)
        self.sum_absolute_error += torch.sum(absolute_error * weights)
        self.sum_weights += torch.sum(weights)

    def compute(self):
        return self.sum_absolute_error / self.sum_weights
