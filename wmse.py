# torchmetrics/functional/regression/wmse.py
import torch
from torchmetrics import Metric


class WeightedMeanSquaredError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("sum_weights", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        squared_error = (preds - target) ** 2
        self.sum_squared_error += torch.sum(squared_error * weights)
        self.sum_weights += torch.sum(weights)

    def compute(self):
        return self.sum_squared_error / self.sum_weights
