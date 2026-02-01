# PyTorch Deep Learning Cheatsheet

Quick reference for common PyTorch operations and deep learning patterns.

---

## Table of Contents

1. [Tensor Operations](#tensor-operations)
2. [GPU Operations](#gpu-operations)
3. [Autograd](#autograd)
4. [Neural Network Layers](#neural-network-layers)
5. [Activation Functions](#activation-functions)
6. [Loss Functions](#loss-functions)
7. [Optimizers](#optimizers)
8. [Data Loading](#data-loading)
9. [Training Loop Template](#training-loop-template)
10. [Model Save/Load](#model-saveload)
11. [Common Patterns](#common-patterns)

---

## Tensor Operations

### Creating Tensors

```python
import torch

# From Python lists
x = torch.tensor([1, 2, 3])
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# Special tensors
zeros = torch.zeros(3, 4)           # 3x4 zeros
ones = torch.ones(3, 4)             # 3x4 ones
eye = torch.eye(3)                  # 3x3 identity
empty = torch.empty(3, 4)           # Uninitialized
rand = torch.rand(3, 4)             # Uniform [0, 1)
randn = torch.randn(3, 4)           # Normal(0, 1)
randint = torch.randint(0, 10, (3, 4))  # Random integers

# From NumPy
import numpy as np
np_arr = np.array([1, 2, 3])
x = torch.from_numpy(np_arr)        # Shares memory!
x = torch.tensor(np_arr)            # Copies data

# Like another tensor
y = torch.zeros_like(x)
y = torch.ones_like(x)
y = torch.randn_like(x)
```

### Tensor Attributes

```python
x.shape          # Size of tensor
x.dtype          # Data type
x.device         # CPU or CUDA
x.requires_grad  # Track gradients?
x.ndim           # Number of dimensions
x.numel()        # Total number of elements
```

### Reshaping

```python
x.view(2, 6)          # Reshape (must be contiguous)
x.reshape(2, 6)       # Reshape (works always)
x.squeeze()           # Remove dimensions of size 1
x.unsqueeze(0)        # Add dimension at position 0
x.flatten()           # Flatten to 1D
x.permute(2, 0, 1)    # Reorder dimensions
x.transpose(0, 1)     # Swap two dimensions
x.T                   # Transpose (2D only)
```

### Indexing and Slicing

```python
x[0]              # First element
x[-1]             # Last element
x[1:3]            # Elements 1 and 2
x[:, 0]           # First column
x[x > 0]          # Boolean indexing
x.masked_fill(mask, value)  # Fill where mask is True
```

### Math Operations

```python
# Element-wise
x + y, x - y, x * y, x / y
x ** 2, torch.sqrt(x), torch.exp(x), torch.log(x)
torch.abs(x), torch.clamp(x, min=0, max=1)

# Reduction
x.sum(), x.mean(), x.std(), x.var()
x.min(), x.max(), x.argmin(), x.argmax()
x.sum(dim=0)      # Sum along dimension 0

# Matrix operations
torch.dot(x, y)           # Dot product (1D)
torch.matmul(A, B)        # Matrix multiplication
A @ B                     # Same as matmul
torch.mm(A, B)            # Matrix mult (2D only)
torch.bmm(A, B)           # Batch matrix mult
```

---

## GPU Operations

```python
# Check availability
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.current_device()

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
x = x.cuda()              # Explicitly to GPU
x = x.cpu()               # Back to CPU

# Create on GPU
x = torch.randn(3, 4, device=device)

# Model to GPU
model = model.to(device)
```

---

## Autograd

```python
# Enable gradient tracking
x = torch.tensor([1.0, 2.0], requires_grad=True)

# Forward pass
y = x ** 2 + 3 * x
loss = y.sum()

# Backward pass
loss.backward()

# Access gradients
x.grad

# Detach from graph
x_detached = x.detach()

# No gradient context
with torch.no_grad():
    # Operations here won't track gradients
    y = model(x)

# Zero gradients
optimizer.zero_grad()
# or
x.grad.zero_()
```

---

## Neural Network Layers

```python
import torch.nn as nn

# Linear (Dense) layer
nn.Linear(in_features, out_features, bias=True)

# Activations
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=1)
nn.LeakyReLU(negative_slope=0.01)

# Regularization
nn.Dropout(p=0.5)
nn.BatchNorm1d(num_features)

# Sequential container
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)
```

### Custom Module

```python
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MyModel(784, 256, 10)
```

---

## Activation Functions

| Function | Usage | Output Range |
|----------|-------|--------------|
| `nn.Sigmoid()` | Binary classification output | (0, 1) |
| `nn.Tanh()` | Hidden layers (older) | (-1, 1) |
| `nn.ReLU()` | Hidden layers (default) | [0, inf) |
| `nn.LeakyReLU(0.01)` | Hidden layers | (-inf, inf) |
| `nn.Softmax(dim=1)` | Multi-class output | (0, 1), sum=1 |
| `nn.LogSoftmax(dim=1)` | With NLLLoss | (-inf, 0) |

**Functional versions:**
```python
import torch.nn.functional as F
F.relu(x)
F.sigmoid(x)
F.softmax(x, dim=1)
```

---

## Loss Functions

```python
# Regression
nn.MSELoss()                  # Mean Squared Error
nn.L1Loss()                   # Mean Absolute Error
nn.SmoothL1Loss()             # Huber Loss

# Binary Classification
nn.BCELoss()                  # Binary Cross Entropy (after sigmoid)
nn.BCEWithLogitsLoss()        # BCE + Sigmoid (more stable)

# Multi-class Classification
nn.CrossEntropyLoss()         # Softmax + NLL (use raw logits)
nn.NLLLoss()                  # Negative Log Likelihood (after log_softmax)

# Usage
criterion = nn.CrossEntropyLoss()
loss = criterion(predictions, targets)
```

**Quick Selection:**
| Task | Loss Function | Output Activation |
|------|---------------|-------------------|
| Regression | MSELoss | None |
| Binary Classification | BCEWithLogitsLoss | None (raw logits) |
| Multi-class | CrossEntropyLoss | None (raw logits) |

---

## Optimizers

```python
import torch.optim as optim

# Common optimizers
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# Learning rate schedulers
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# In training loop
scheduler.step()              # Call after epoch
scheduler.step(val_loss)      # For ReduceLROnPlateau
```

---

## Data Loading

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Simple dataset from tensors
dataset = TensorDataset(X_tensor, y_tensor)

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,          # Shuffle for training
    num_workers=4,         # Parallel loading
    pin_memory=True,       # Faster GPU transfer
    drop_last=True         # Drop incomplete final batch
)

# Iterate
for batch_x, batch_y in loader:
    # Process batch
    pass
```

---

## Training Loop Template

```python
# Setup
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Training mode
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()  # Evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Val Acc: {100.*correct/total:.2f}%')
```

---

## Model Save/Load

```python
# Save entire model
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# Save state dict only (recommended)
torch.save(model.state_dict(), 'model_weights.pth')
model.load_state_dict(torch.load('model_weights.pth'))

# Save checkpoint (for resuming training)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

## Common Patterns

### Weight Initialization

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)

# Individual layer
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(layer.bias)
```

### Freeze Layers

```python
# Freeze all
for param in model.parameters():
    param.requires_grad = False

# Unfreeze specific layer
for param in model.fc.parameters():
    param.requires_grad = True
```

### Model Summary

```python
# Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total: {total:,}, Trainable: {trainable:,}')
```

### Reproducibility

```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## Quick Reference Cards

### Dimension Guide

| Operation | Input Shape | Output Shape |
|-----------|-------------|--------------|
| `nn.Linear(in, out)` | (*, in) | (*, out) |
| `nn.BatchNorm1d(n)` | (batch, n) | (batch, n) |
| `nn.Dropout(p)` | Any | Same |
| `x.view(-1)` | Any | (numel,) |
| `x.unsqueeze(0)` | (a, b) | (1, a, b) |
| `x.squeeze()` | (1, a, 1, b) | (a, b) |

### Common Errors and Fixes

| Error | Likely Cause | Fix |
|-------|--------------|-----|
| Size mismatch | Wrong tensor shapes | Check dimensions |
| CUDA error | Mixed CPU/GPU tensors | Move all to same device |
| Gradient None | Not part of graph | Check requires_grad |
| NaN loss | Exploding gradients | Lower learning rate |
| Memory error | Batch too large | Reduce batch size |
