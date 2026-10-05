# numpy_grad

A small neural network library built from scratch in pure NumPy. It implements fully connected networks, ReLU, softmax cross-entropy, backpropagation, and gradient descent by hand, with no autograd and no deep learning framework.

Written to understand how training actually works, and structured so that models, optimizers, and the training loop are separate, swappable pieces.

## Features

- Dense layers with ReLU and He initialization
- Hand-written backpropagation (per-sample and fully vectorized batched versions)
- Softmax cross-entropy loss with a shared, numerically stable softmax
- Modular design: `Model`, `Optimizer`, and `Trainer` are independent, so a new optimizer never needs its own training loop
- In-place parameter updates through a single `parameters()` interface
- float32 throughout, with preallocated buffers to limit allocations

## Results

A 196-121-64-25-10 MLP trained with plain SGD on MNIST downsampled to 14x14 (2x2 average pooling) reaches about **97.8% test accuracy** (test loss ~0.07).

## Install

From the project root (the folder containing `numpy_grad/`), the package works directly when you run code from that directory. To install it so it works from anywhere, put `pyproject.toml` next to the `numpy_grad/` folder with `packages = ["numpy_grad"]` and run:

```
pip install -e .
```

Requires Python 3.9+ and NumPy.

## Quick start

```python
import numpy as np
from numpy_grad import MLP, SGD, Trainer

# X_train: (N, 196) float32, Y_train: (N, 10) one-hot float32
model = MLP(nin=196, nouts=[121, 64, 25, 10])
optimizer = SGD(model, learning_rate=0.00005)

trainer = Trainer(optimizer, X_train, Y_train, batching=True, batch_size=16)
trainer.train(n_steps=len(X_train) // 16, report_freq=500)

# evaluate
z = model.forward_batch(X_test)
print("accuracy:", (z.argmax(axis=1) == Y_test.argmax(axis=1)).mean())
```

## Design

```
Trainer  ->  Optimizer  ->  Model  ->  Layer
 (loop)      (update rule)  (forward/   (weights, gradients)
                             backward)
```

- **Layer**: owns its weights, biases, and gradient buffers.
- **Model**: composes layers; exposes `parameters()` as `(param, grad)` pairs.
- **Optimizer**: reads `parameters()` and updates the arrays in place.
- **Trainer**: runs the loop; works with any optimizer.

Two execution paths compute the same math:

| Mode | Trainer args | Description |
|---|---|---|
| Chunking | `batching=False, chunk_size=N` | one sample at a time, gradients summed over N samples |
| Batching | `batching=True, batch_size=N` | the N samples go through as one `(N, nin)` matrix |

Gradients are summed over each group, so scale the learning rate when you change the group size.

## Adding an optimizer

```python
class Momentum(Optimizer):
    def __init__(self, model, learning_rate, beta=0.9):
        self.model, self.lr, self.beta = model, learning_rate, beta
        self.v = [np.zeros_like(p) for p, _ in model.parameters()]

    def step_accumulate_gradients(self, x, y): ...  # same as SGD
    def step_accumulate_batch(self, x, y): ...      # same as SGD

    def step_learn(self):
        for v, (p, g) in zip(self.v, self.model.parameters()):
            v *= self.beta
            v += g
            p -= self.lr * v
        self.model.zero_grad()
```

## Documentation

See [DOCS.md](DOCS.md) for the full API reference.

## Limitations

- Dense layers and ReLU only
- Softmax cross-entropy is the only loss
- Not thread-safe (layers cache activations from the last forward pass)

## License

MIT
