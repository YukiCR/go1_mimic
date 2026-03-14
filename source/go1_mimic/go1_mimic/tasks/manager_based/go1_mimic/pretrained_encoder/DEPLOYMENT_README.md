# Go1 Autoencoder - TorchScript Deployment Guide

## Model Files

- `go1_autoencoder_full.pt`: Full autoencoder
- `go1_autoencoder_encoder.pt`: Encoder autoencoder
- `go1_autoencoder_decoder.pt`: Decoder autoencoder

## Input/Output Specifications

### Full Autoencoder
- Input: `[B, 1, 64, 64]`
- Output: `[B, 1, 64, 64]`

### Encoder Encoder
- Input: `[B, 1, 64, 64]`
- Output: `[B, 64]`

### Decoder Decoder
- Input: `[B, 64]`
- Output: `[B, 1, 64, 64]`

## Preprocessing Requirements

**Input Normalization** (must be applied before forward):
```python
mean = [1.77]
std = [2.58]
normalized = (input - mean) / std
```

Where input is depth in meters (0.0 = infinity/no return).

## Usage Example

```python
import torch

# Load without model definition
model = torch.jit.load('go1_autoencoder_encoder.pt')
model.eval()

# Prepare input (example: batch_size=1)
input_tensor = torch.randn(1, 1, 64, 64)

# Forward pass
with torch.no_grad():
    latent = model(input_tensor)  # [1, 64]
```

## Metadata
- Exported: 2.7.0+cu128
- Checkpoint: best_model.pt
- Config: config.yaml
- Device: cpu
