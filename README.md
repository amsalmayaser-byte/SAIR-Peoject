Custom Neural Network Library (NumPy) 🧠💻
Version 1.0 – First Release 🚀

A lightweight and modular neural network library implemented from scratch using Python and NumPy.
This project is designed to deeply understand and implement the internal mechanics of neural networks — including forward propagation, backpropagation, activation functions, optimizers, and custom loss functions — without relying on frameworks like PyTorch or TensorFlow.
This library prioritizes clarity, correctness, and educational value over computational performance.

🎯 Project Goals

Understand how neural networks work internally at a low level
Implement backpropagation manually using the chain rule
Design a modular architecture similar to modern deep learning libraries
Explore optimization techniques such as Adam
Integrate regularization to control overfitting
Create custom loss functions and visualize outputs including matrices

🚀 Key Features

Modular Design – Layers are independent and reusable
Dynamic Networks – Build networks by stacking layers
Extensible – Easily add new layers, activation functions, or loss functions
Custom Loss Functions – Fully implemented with plotting support
Visualization Support – Plot outputs, weights, and matrices
🔌 Activation Functions
Implemented with forward and backward passes:

ReLU
Sigmoid
Tanh

🛡 Elastic Net Regularization (L1 & L2)
L1 Regularization encourages sparsity in weights
L2 Regularization penalizes large weights to improve numerical stability
Regularization terms are added directly to the weight gradients during backpropagation. Bias terms are not regularized.

⚙️ Optimizers
SGD (baseline optimizer)
Adam, featuring:
First moment (momentum) estimation
Second moment (adaptive scaling)
Bias correction
Independent optimizer state per layer

📝 Design & Mathematical Notes
Gradients are computed manually (no automatic differentiation)
All tensors are handled explicitly as NumPy arrays with fixed shapes
Each layer caches its forward-pass inputs for use during backpropagation
Optimizer state (moments, time step) is explicitly managed
Regularization is applied only to weight matrices
This design mimics the internal mechanics of modern deep learning frameworks while remaining fully transparent.

📂 Project Structure

├── confusion_matrix_heatmap.png
├── loss_curve_batch.png
├── nn_library
│   ├── activations
│   │   ├── functions.py
│   │   ├── _init_.py
│   │   └── _pycache_
│   │       ├── functions.cpython-312.pyc
│   │       └── _init_.cpython-312.pyc
│   ├── core
│   │   ├── _init_.py
│   │   ├── layer.py
│   │   ├── network.py
│   │   ├── neuron.py
│   │   └── _pycache_
│   │       ├── _init_.cpython-312.pyc
│   │       ├── layer.cpython-312.pyc
│   │       └── network.cpython-312.pyc
│   ├── losses
│   │   ├── functions.py
│   │   ├── _init_.py
│   │   └── _pycache_
│   │       ├── functions.cpython-312.pyc
│   │       └── _init_.cpython-312.pyc
│   ├── optimizers
│   │   ├── _init_.py
│   │   ├── optimizers.py
│   │   └── _pycache_
│   │       ├── _init_.cpython-312.pyc
│   │       └── optimizers.cpython-312.pyc
│   ├── start.py
│   └── utils
│       ├── batch_generator.py
│       ├── _intt_.py
│       └── _pycache_
│           └── batch_generator.cpython-312.pyc
├── start.py
└── train_val_loss_curve.png
