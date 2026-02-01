# PyTorch and Deep Learning Fundamentals

A comprehensive revision repository structured for both **learning** and **interview preparation**. This repository covers deep learning fundamentals from mathematical prerequisites through complete Artificial Neural Networks (ANN).

---

## Scope

This repository covers **fundamentals up to ANN only**:
- Mathematical foundations
- PyTorch basics
- Neural network building blocks
- Training and optimization

**Not included** (covered in separate repository):
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Transformers
- Advanced architectures

---

## Prerequisites

Before starting, you should have:
- Python 3.8+ installed
- Basic Python programming knowledge
- Jupyter Notebook/Lab or VS Code with Jupyter extension
- PyTorch installed (`pip install torch torchvision`)
- NumPy, Matplotlib, Seaborn installed

---

## How to Use This Repository

1. **Follow the modules in order** - Each module builds on previous concepts
2. **Run every code cell** - Understanding comes from doing
3. **Complete the exercises** - Practice solidifies learning
4. **Review interview tips** - Prepare for technical interviews
5. **Use the cheatsheets** - Quick reference during revision

---

## Learning Path

### Foundation (Start Here)

| Module | Topic | Description |
|--------|-------|-------------|
| [01](01_python_math_prerequisites/01_prerequisites.ipynb) | Python & Math Prerequisites | NumPy, calculus, linear algebra, probability |
| [02](02_intro_to_deep_learning/02_intro_deep_learning.ipynb) | Introduction to Deep Learning | DL vs ML, history, why it works, hardware |
| [03](03_pytorch_fundamentals/03_pytorch_fundamentals.ipynb) | PyTorch Fundamentals | Tensors, GPU, autograd basics |

### Building Blocks

| Module | Topic | Description |
|--------|-------|-------------|
| [04](04_the_neuron/04_neuron.ipynb) | The Neuron | Biological vs artificial, mathematical model |
| [05](05_activation_functions/05_activation_functions.ipynb) | Activation Functions | Sigmoid, tanh, ReLU, softmax and variants |
| [06](06_perceptron/06_perceptron.ipynb) | Perceptron | Single layer, learning algorithm, XOR problem |

### Learning Mechanics

| Module | Topic | Description |
|--------|-------|-------------|
| [07](07_loss_functions/07_loss_functions.ipynb) | Loss Functions | MSE, cross-entropy, when to use which |
| [08](08_gradient_descent/08_gradient_descent.ipynb) | Gradient Descent | Batch, stochastic, mini-batch variants |
| [09](09_backpropagation/09_backpropagation.ipynb) | Backpropagation | Chain rule, forward/backward pass, autograd |

### Optimization

| Module | Topic | Description |
|--------|-------|-------------|
| [10](10_optimizers/10_optimizers.ipynb) | Optimizers | SGD momentum, Adam, learning rate scheduling |
| [11](11_building_neural_networks/11_nn_pytorch.ipynb) | Building NNs in PyTorch | nn.Module, layers, custom networks |
| [12](12_training_pipeline/12_training_pipeline.ipynb) | Training Pipeline | DataLoader, training loop, checkpointing |

### Advanced Techniques

| Module | Topic | Description |
|--------|-------|-------------|
| [13](13_regularization/13_regularization.ipynb) | Regularization | L1/L2, dropout, early stopping |
| [14](14_weight_initialization/14_weight_initialization.ipynb) | Weight Initialization | Xavier, He, why it matters |
| [15](15_batch_normalization/15_batch_normalization.ipynb) | Batch Normalization | Internal covariate shift, train vs eval |
| [16](16_hyperparameter_tuning/16_hyperparameter_tuning.ipynb) | Hyperparameter Tuning | Grid search, random search, best practices |

### Putting It Together

| Module | Topic | Description |
|--------|-------|-------------|
| [17](17_practical_project/17_complete_ann_project.ipynb) | Practical Project | End-to-end classification project |
| [18](18_practice_problems/18_interview_prep.ipynb) | Practice & Interview Prep | Questions, exercises, derivations |

---

## Quick References

- [CHEATSHEET.md](CHEATSHEET.md) - PyTorch operations quick reference
- [cheatsheet.ipynb](cheatsheet.ipynb) - Interactive cheatsheet with runnable code
- [math_notation.md](math_notation.md) - Mathematical symbols reference

---

## Notebook Format

Each notebook follows a consistent structure:

1. **Objectives** - What you will learn
2. **Intuition** - Conceptual understanding without code
3. **Mathematical Foundation** - Equations with explanations
4. **Code from Scratch** - NumPy implementation
5. **PyTorch Implementation** - Using the framework properly
6. **Visualizations** - Graphs and plots
7. **Key Points Summary** - Quick revision bullets
8. **Interview Tips** - Common questions and answers
9. **Practice Exercises** - Problems to solve

---

## Sample Datasets

The [datasets/](datasets/) folder contains sample data for exercises. See [datasets/README.md](datasets/README.md) for details.

---

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch torchvision numpy matplotlib seaborn pandas scikit-learn jupyter
```

---

## Progress Tracker

Use this to track your learning:

- [ ] Module 01: Python & Math Prerequisites
- [ ] Module 02: Introduction to Deep Learning
- [ ] Module 03: PyTorch Fundamentals
- [ ] Module 04: The Neuron
- [ ] Module 05: Activation Functions
- [ ] Module 06: Perceptron
- [ ] Module 07: Loss Functions
- [ ] Module 08: Gradient Descent
- [ ] Module 09: Backpropagation
- [ ] Module 10: Optimizers
- [ ] Module 11: Building Neural Networks
- [ ] Module 12: Training Pipeline
- [ ] Module 13: Regularization
- [ ] Module 14: Weight Initialization
- [ ] Module 15: Batch Normalization
- [ ] Module 16: Hyperparameter Tuning
- [ ] Module 17: Practical Project
- [ ] Module 18: Practice & Interview

---

## Contributing

This is a personal revision repository. Feel free to fork and adapt for your own learning journey.

---

## License

MIT License - Feel free to use for learning purposes.
