import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from wmae import WeightedMeanAbsoluteError
from wmse import WeightedMeanSquaredError

# Simple dataset with weights
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
weights = torch.tensor([1.0, 0.5, 2.0, 1.5])

dataset = TensorDataset(x, y, weights)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = SimpleModel()
criterion = nn.MSELoss(reduction="none")  # Use reduction='none' to get per-sample loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Initialize metrics
wmse_metric = WeightedMeanSquaredError()
wmae_metric = WeightedMeanAbsoluteError()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        inputs, targets, batch_weights = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        weighted_loss = loss * batch_weights.view(-1, 1)
        weighted_loss.mean().backward()
        optimizer.step()

        # Update custom metrics
        wmse_metric.update(outputs, targets, batch_weights)
        wmae_metric.update(outputs, targets, batch_weights)

    # Compute metrics at the end of each epoch
    wmse = wmse_metric.compute()
    wmae = wmae_metric.compute()
    print(f"Epoch {epoch+1}/{num_epochs}, WMSE: {wmse.item()}, WMAE: {wmae.item()}")

    # Reset metrics for the next epoch
    wmse_metric.reset()
    wmae_metric.reset()
