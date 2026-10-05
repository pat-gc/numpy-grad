# numpy_grad

A small neural network library written in pure NumPy. It supports fully connected networks (MLPs) with ReLU activations, softmax cross-entropy loss, and gradient descent training. Training can run per sample (chunking) or vectorized (batching).

## Layout

```
numpy_grad/
├── __init__.py      # re-exports the public names
└── numpy_grad.py    # all the code
```

## Quick start

```python
import numpy as np
from numpy_grad import MLP, SGD, Trainer

# X_train: (N, 196) float32, Y_train: (N, 10) one-hot float32
model = MLP(nin=196, nouts=[121, 64, 25, 10])
optimizer = SGD(model, learning_rate=0.00005)

trainer = Trainer(optimizer, X_train, Y_train, batching=True, batch_size=16)
trainer.train(n_steps=len(X_train) // 16, report_freq=500)
```

Evaluate on the whole test set in one call:

```python
z = model.forward_batch(X_test)
accuracy = (z.argmax(axis=1) == Y_test.argmax(axis=1)).mean()
```

## Data expectations

- Inputs: `float32` arrays, one flattened row per sample, shape `(N, nin)`.
- Labels: one-hot `float32`, shape `(N, nclasses)`.
- At least 2 output classes (softmax gradient is meaningless with 1).

## Concepts

**Training step.** One step accumulates gradients over a group of samples, then applies one weight update (`step_learn`) and clears the gradients. Gradients are **summed** over the group, not averaged, so the effective step size grows with the group size. Scale the learning rate accordingly when changing it.

**Chunking vs batching.** Both compute the same math.

| Mode | Trainer args | Path | Group size variable |
|---|---|---|---|
| Chunking | `batching=False` | one sample at a time, loop in Python | `chunk_size` |
| Batching | `batching=True` | whole group as a `(B, nin)` matrix | `batch_size` |

Batching is much faster because the NumPy call overhead is paid once per group instead of once per sample.

## API reference

### `Layer(nin, nout, nonlin, seed=42)`

One fully connected layer. Each of the `nout` rows of `weights` is a neuron.

| Attribute | Shape | Meaning |
|---|---|---|
| `weights` | `(nout, nin)` | He-initialized weights |
| `bias_weights` | `(nout,)` | biases |
| `gradients` | `(nout, nin)` | accumulated weight gradients |
| `bias_gradients` | `(nout,)` | accumulated bias gradients |
| `mask` | `(nout,)` or `(B, nout)` | ReLU mask from the last forward pass |
| `_tmp` | `(nout, nin)` | reusable buffer to avoid allocation |

`nonlin=True` applies ReLU; `False` leaves the layer linear.

Methods:

- `__call__(x)`: forward pass for one sample `(nin,)`.
- `forward_batch(x)`: forward pass for a batch `(B, nin) -> (B, nout)`.
- `accumulate_gradients(g)`: backward pass for one sample. Adds into `gradients` and `bias_gradients`, returns the gradient for the layer input.
- `accumulate_gradients_batch(g)`: same for a batch `(B, nout) -> (B, nin)`.
- `zero_grad()`: resets the gradient buffers.
- `parameters()`: yields `(weights, gradients)` then `(bias_weights, bias_gradients)`.

A backward call must follow its matching forward call, because the layer stores the input and mask from the last forward pass.

### `Model` (abstract)

The interface optimizers depend on. Subclasses must implement:

- `__call__(x)`, `forward_batch(x)`
- `accumulate_gradients(g)`, `accumulate_gradients_batch(g)`
- `parameters()`: yields `(param, grad)` pairs of the real arrays (references, not copies)

`zero_grad()` is provided and calls `zero_grad()` on each layer in `self.layers`.

### `MLP(nin, nouts)`

A stack of `Layer`s. `nouts` lists the width of each layer, for example `[121, 64, 25, 10]`. All layers use ReLU except the last, which is linear (it outputs logits).

- `parameters()` flattens all layers into one sequence of `(param, grad)` pairs, 2 per layer.
- `stats()` prints a table of layers, parameter counts, and memory use.
- `print(model)` shows every weight, bias, and mask.

### `Optimizer` (abstract)

Subclasses implement:

- `step_accumulate_gradients(x, y) -> float`: one sample, returns its loss.
- `step_accumulate_batch(x, y) -> float`: a batch, returns the summed loss.
- `step_learn()`: apply the update and clear gradients.

### `SGD(model, learning_rate)`

Plain gradient descent. `step_learn` does `p -= lr * g` for every parameter pair, in place, then calls `model.zero_grad()`. Note that it scales `g` in place, so do not read gradients after `step_learn`.

### `Trainer(optimizer, x_train_set, y_train_set, batching=False, chunk_size=-1, batch_size=-1)`

Drives the training loop.

- `train(n_steps, report_freq=1)`: runs `n_steps` updates, each using the next `size` samples (`batch_size` if batching, else `chunk_size`). Prints the mean loss every `report_freq` steps.
- One epoch is `n_steps = len(X_train) // size`.
- The loop does not wrap around: `n_steps * size` must not exceed the dataset length. Shuffle the data beforehand.

### Loss functions

All use softmax cross-entropy on raw logits `z` and one-hot `y`.

- `z_loss_and_gradient(z, y) -> (loss, grad)`: single sample, shares one softmax for both.
- `z_loss_and_gradient_batch(z, y) -> (loss, grad)`: `(B, nclasses)` inputs, loss summed over the batch.
- `z_loss(z, y)`: loss only, for evaluation.

The gradient is `softmax(z) - y`.

## Extending

**New optimizer.** Subclass `Optimizer` and use `model.parameters()`. Keep per-parameter state in a list aligned with the pairs, created once:

```python
class Momentum(Optimizer):
    def __init__(self, model, learning_rate, beta=0.9):
        self.model, self.lr, self.beta = model, learning_rate, beta
        self.v = [np.zeros_like(p) for p, _ in model.parameters()]

    def step_accumulate_gradients(self, x, y): ...   # same as SGD
    def step_accumulate_batch(self, x, y): ...       # same as SGD

    def step_learn(self):
        for v, (p, g) in zip(self.v, self.model.parameters()):
            v *= self.beta
            v += g
            p -= self.lr * v
        self.model.zero_grad()
```

**New model.** Subclass `Model`, set `self.layers`, and implement the abstract methods. Optimizers only need `parameters()` and the accumulate methods.

## Notes and known limits

- Only dense layers and ReLU are implemented.
- Gradients are summed over a group, not averaged.
- Weight init uses He initialization (`sqrt(2 / nin)`) for both weights and biases. Biases are often initialized to zero instead.
- Not thread-safe: layers store activations from the last forward pass.

*Note:
Docs are fully ai generated from the code