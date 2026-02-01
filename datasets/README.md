# Sample Datasets

This folder contains sample datasets used throughout the notebooks for exercises and demonstrations.

---

## Included Datasets

### 1. Simple Classification Data

Generated synthetic datasets for learning classification concepts:
- Binary classification examples
- Multi-class classification examples
- Linearly separable and non-linearly separable data

### 2. Regression Data

Simple datasets for regression exercises:
- Linear relationships
- Non-linear relationships
- Multi-variable examples

---

## Loading Datasets

### Using PyTorch Built-in Datasets

Most notebooks use PyTorch's built-in datasets for convenience:

```python
from torchvision import datasets, transforms

# MNIST (handwritten digits)
train_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Fashion-MNIST
train_data = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
```

### Using scikit-learn Datasets

For simpler examples:

```python
from sklearn.datasets import make_classification, make_regression, make_moons

# Binary classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)

# Non-linear classification
X, y = make_moons(n_samples=1000, noise=0.1)

# Regression
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
```

### Converting to PyTorch

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # For classification
# y_tensor = torch.tensor(y, dtype=torch.float32)  # For regression

# Create dataset and loader
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## Data Preprocessing Tips

### Normalization

```python
# Min-Max scaling to [0, 1]
X_normalized = (X - X.min()) / (X.max() - X.min())

# Standardization (zero mean, unit variance)
X_standardized = (X - X.mean()) / X.std()
```

### Train/Validation/Test Split

```python
from sklearn.model_selection import train_test_split

# 60% train, 20% val, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
```

---

## Attribution

Some datasets used in this repository come from:
- PyTorch/torchvision built-in datasets
- scikit-learn generated datasets
- UCI Machine Learning Repository

All datasets are used for educational purposes only.
