# Mathematical Notation Reference

This document provides a reference for all mathematical symbols and notation used throughout this repository.

---

## Greek Letters

| Symbol | Name | Common Usage |
|--------|------|--------------|
| α (alpha) | Alpha | Learning rate |
| β (beta) | Beta | Momentum coefficient, bias term |
| γ (gamma) | Gamma | Batch norm scale parameter |
| δ (delta) | Delta | Error term, small change |
| ε (epsilon) | Epsilon | Small constant (numerical stability) |
| η (eta) | Eta | Learning rate (alternative) |
| θ (theta) | Theta | Model parameters (general) |
| λ (lambda) | Lambda | Regularization strength |
| μ (mu) | Mu | Mean |
| σ (sigma) | Sigma | Standard deviation, sigmoid function |
| Σ (Sigma) | Capital Sigma | Summation |
| φ (phi) | Phi | Activation function (general) |
| ψ (psi) | Psi | Alternative activation notation |
| ω (omega) | Omega | Frequency, alternative weight notation |

---

## Operators and Symbols

### Basic Operations

| Symbol | Meaning | Example |
|--------|---------|---------|
| + | Addition | a + b |
| - | Subtraction | a - b |
| × or · | Multiplication | a × b or a · b |
| / or ÷ | Division | a / b |
| ^ or superscript | Exponentiation | x^2 or x² |
| √ | Square root | √x |
| \| \| | Absolute value | \|x\| |

### Summation and Products

| Symbol | Meaning | Example |
|--------|---------|---------|
| Σ | Sum | Σᵢ xᵢ = x₁ + x₂ + ... + xₙ |
| Π | Product | Πᵢ xᵢ = x₁ × x₂ × ... × xₙ |

### Calculus

| Symbol | Meaning | Example |
|--------|---------|---------|
| d/dx | Derivative with respect to x | df/dx |
| ∂/∂x | Partial derivative | ∂f/∂x |
| ∇ | Gradient (nabla) | ∇f = [∂f/∂x₁, ∂f/∂x₂, ...] |
| ∫ | Integral | ∫f(x)dx |

### Set and Logic

| Symbol | Meaning | Example |
|--------|---------|---------|
| ∈ | Element of | x ∈ R (x is a real number) |
| ∀ | For all | ∀x (for all x) |
| ∃ | There exists | ∃x (there exists x) |
| ⊂ | Subset | A ⊂ B |
| ∩ | Intersection | A ∩ B |
| ∪ | Union | A ∪ B |

### Comparison

| Symbol | Meaning |
|--------|---------|
| = | Equal to |
| ≠ | Not equal to |
| < | Less than |
| > | Greater than |
| ≤ | Less than or equal to |
| ≥ | Greater than or equal to |
| ≈ | Approximately equal |
| ∝ | Proportional to |

---

## Linear Algebra Notation

### Vectors

| Notation | Meaning |
|----------|---------|
| **x** or x (bold) | Vector |
| xᵢ | i-th element of vector x |
| x^T | Transpose of vector x |
| \|\|x\|\| | Norm (magnitude) of vector x |
| \|\|x\|\|₂ | L2 norm (Euclidean) |
| \|\|x\|\|₁ | L1 norm (Manhattan) |

### Matrices

| Notation | Meaning |
|----------|---------|
| **W** or W (capital bold) | Matrix |
| Wᵢⱼ | Element at row i, column j |
| W^T | Transpose of matrix W |
| W⁻¹ | Inverse of matrix W |
| det(W) | Determinant of W |
| tr(W) | Trace of W (sum of diagonal) |
| I | Identity matrix |

### Operations

| Notation | Meaning |
|----------|---------|
| **x** · **y** | Dot product |
| **x** × **y** | Cross product |
| **A** **B** | Matrix multiplication |
| **A** ⊙ **B** | Element-wise (Hadamard) product |
| **A** ⊗ **B** | Kronecker product |

---

## Neural Network Specific

### Layers and Neurons

| Symbol | Meaning |
|--------|---------|
| L | Total number of layers |
| l | Layer index (superscript) |
| n^[l] | Number of neurons in layer l |
| a^[l] | Activation of layer l |
| z^[l] | Pre-activation (weighted sum) of layer l |
| W^[l] | Weight matrix for layer l |
| b^[l] | Bias vector for layer l |

### Activations

| Symbol | Function |
|--------|----------|
| σ(z) | Sigmoid: 1/(1+e^(-z)) |
| tanh(z) | Hyperbolic tangent |
| ReLU(z) | max(0, z) |
| softmax(z)ᵢ | e^(zᵢ) / Σⱼ e^(zⱼ) |

### Loss Functions

| Symbol | Function |
|--------|----------|
| L or J | Loss/Cost function |
| ŷ | Predicted output |
| y | True output |
| MSE | Mean Squared Error: (1/n)Σ(y-ŷ)² |
| BCE | Binary Cross-Entropy |
| CE | Categorical Cross-Entropy |

### Training

| Symbol | Meaning |
|--------|---------|
| m | Number of training examples |
| n | Number of features |
| X | Input data matrix (m × n) |
| Y | Output data matrix |
| α or η | Learning rate |
| t | Time step / iteration |
| θₜ | Parameters at time t |

---

## Probability and Statistics

| Symbol | Meaning |
|--------|---------|
| P(A) | Probability of event A |
| P(A\|B) | Conditional probability of A given B |
| E[X] | Expected value of X |
| Var(X) | Variance of X |
| Cov(X,Y) | Covariance of X and Y |
| N(μ,σ²) | Normal distribution with mean μ, variance σ² |
| U(a,b) | Uniform distribution between a and b |

---

## Subscript and Superscript Conventions

### Subscripts

| Usage | Meaning | Example |
|-------|---------|---------|
| i, j, k | Index variables | xᵢ (i-th element) |
| t | Time step | θₜ (params at step t) |
| batch | Batch-related | μ_batch |

### Superscripts

| Usage | Meaning | Example |
|-------|---------|---------|
| [l] | Layer number | W^[2] (layer 2 weights) |
| (i) | Training example | x^(i) (i-th example) |
| T | Transpose | W^T |
| -1 | Inverse | W^(-1) |
| * | Optimal value | θ* (optimal params) |

---

## Common Expressions in Deep Learning

### Forward Pass
```
z^[l] = W^[l] a^[l-1] + b^[l]
a^[l] = g^[l](z^[l])
```

### Gradient Descent Update
```
θ = θ - α ∇J(θ)
```

### Chain Rule
```
∂L/∂w = (∂L/∂a) × (∂a/∂z) × (∂z/∂w)
```

### Backpropagation
```
δ^[l] = (W^[l+1])^T δ^[l+1] ⊙ g'^[l](z^[l])
∂J/∂W^[l] = δ^[l] (a^[l-1])^T
```

---

## Tips for Reading Equations

1. **Identify the variables first** - What does each symbol represent?
2. **Check dimensions** - Matrix operations require compatible dimensions
3. **Look for patterns** - Many equations follow similar structures
4. **Trace the computation** - Follow data flow from input to output
5. **Simplify to scalars** - If confused, think of the scalar case first
