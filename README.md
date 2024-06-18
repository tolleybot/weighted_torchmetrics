# Custom Metrics for PyTorch

This repository provides implementations of custom metrics `WeightedMeanSquaredError` (WMSE) and `WeightedMeanAbsoluteError` (WMAE) for use in PyTorch models.

## Requirements

- Python 3.6+
- PyTorch
- TorchMetrics


## Usage

```python
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
```