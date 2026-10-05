from abc import ABC, abstractmethod
from typing import Iterator
from numpy.typing import NDArray
import numpy as np


    

class Trainer():
    def __init__(self, optimizer: Optimizer, x_train_set: NDArray, y_train_set: NDArray , batching=False, chunk_size=-1, batch_size = -1):
        self.optimizer = optimizer
        self.batching = batching
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.x_train_set = x_train_set
        self.y_train_set = y_train_set

    def train(self, n_steps=-1, report_freq=1):
        size = self.batch_size if self.batching else self.chunk_size
    
        for i in range(n_steps):
            s, e = i * size, (i + 1) * size
            x, y = self.x_train_set[s:e], self.y_train_set[s:e]
    
            if self.batching:
                loss = self.optimizer.step_accumulate_batch(x, y)
            else:
                loss = sum(self.optimizer.step_accumulate_gradients(xi, yi) for xi, yi in zip(x, y))
            self.optimizer.step_learn()
    
            loss /= size
            if i % report_freq == 0: print(f"step: {i} loss = {loss}")



class Model(ABC):

    layers: list

    @abstractmethod
    def accumulate_gradients(self, gradients_forward): pass
    
    def zero_grad(self):
        for layer in self.layers: layer.zero_grad()

    @abstractmethod
    def __call__(self, x): pass

    @abstractmethod
    def parameters(self) -> Iterator[tuple[NDArray[np.float32], NDArray[np.float32]]]: pass

    @abstractmethod
    def forward_batch(self, x): pass

    @abstractmethod
    def accumulate_gradients_batch(self, gradients_forward): pass

    
class Optimizer(ABC):
    
    @abstractmethod
    def step_accumulate_gradients(self, x, y) -> float: pass

    @abstractmethod
    def step_learn(self): pass

    @abstractmethod
    def step_accumulate_batch(self, x, y) -> float: pass



class Layer():
    def __init__(self, nin, nout, nonlin, seed=42): # eah nout is a neuron
        self._tmp = np.zeros((nout, nin), dtype=np.float32) # used as a buffer to avoid reallocating constantly
        self.nout = nout
        self.nin = nin
        rng = np.random.default_rng(seed)

        self.weights = rng.standard_normal((nout, nin), dtype=np.float32) * np.float32(np.sqrt(2.0 / nin)) 
        self.gradients = np.zeros((nout, nin), dtype=np.float32)

        self.bias_weights = rng.standard_normal((nout,), dtype=np.float32) * np.float32(np.sqrt(2.0 / nin)) 
        self.bias_gradients = np.zeros((nout,), dtype=np.float32)

        self.mask = np.ones((nout,), dtype=np.float32)
        self.nonlin = nonlin
        self.activationsin = np.zeros((nin,), dtype=np.float32)
        self.activationsout = np.zeros((nout,), dtype=np.float32)
        

    def __call__(self, activation): # single inference
        self.activationsin = activation
        neurons_out = (self.weights @ activation) + self.bias_weights
        if self.nonlin:
            self.mask = neurons_out > 0 # ReLu activation function
        self.activationsout = neurons_out * self.mask
        return self.activationsout
    

    def forward_batch(self, activation):  # (B, nin) -> (B, nout)
        self.activationsin = activation
        neurons_out = activation @ self.weights.T + self.bias_weights
        if self.nonlin:
            self.mask = neurons_out > 0  # ReLU
        self.activationsout = neurons_out * self.mask
        return self.activationsout


    def accumulate_gradients_batch(self, gradients_forward):  # (B, nout) -> (B, nin)
        g = gradients_forward * self.mask                    # kills dead neurons
        np.matmul(g.T, self.activationsin, out=self._tmp)    # (nout, nin), sum of per-sample outer products
        self.gradients += self._tmp
        self.bias_gradients += g.sum(axis=0)
        return g @ self.weights

    
    def accumulate_gradients(self, gradients_forward):
        # gradients_forward is of shape [nout]
        # self.gradients is of shape [nout, nin]
        # self.activationsin of shape [nin] -> [nout, nin]

        gradients_forward = gradients_forward * self.mask # kills gradients of dead neurons

        # multiplication of 2 1d arrays, gradients forward * activations in of shape (nout, nin)

        np.multiply(gradients_forward[:, np.newaxis], self.activationsin, out=self._tmp)
        self.gradients += self._tmp

        # bias gradient have a local derivative thats always 1 so we only chain to upstream derivative
        self.bias_gradients += gradients_forward

        # gradient at each input, equals the sum of (each weight connected to it) multiplied by its upstream gradient (forward gradient)
        
        gradients_backward = gradients_forward @ self.weights
        return gradients_backward

    
        

    def zero_grad(self):
        self.gradients.fill(0)
        self.bias_gradients.fill(0)

    # def update_weights(self, learning_rate):
    #     np.multiply(learning_rate, self.gradients, out=self._tmp) # weights
        
    #     self.weights -= self._tmp
    #     self.bias_weights -= self.bias_gradients * learning_rate

    def parameters(self):
        yield self.weights, self.gradients
        yield self.bias_weights, self.bias_gradients

    def __repr__(self):
        nin = self.weights.shape[1]
        nout = self.weights.shape[0]
        
        lines = [f"Layer(nin={nin}, nout={nout}, nonlin={self.nonlin}):"]
        
        # Format weights matrix and bias column
        for i in range(nout):
            # Format each weight with fixed total width (e.g., 7 chars, 3 decimals, space for sign)
            w_row = " ".join(f"{w: 7.3f}" for w in self.weights[i])
            bias = f"{self.bias_weights[i]: 7.3f}"
            mask = f"{int(self.mask[i])}"
            
            lines.append(f"  Neuron {i}: W = [{w_row} ]  b = {bias}  mask = {mask}")
            
        return "\n".join(lines)

    
class MLP(Model):
    def __init__(self, nin, nouts): # int, int[]
        self.layers = []

        # making all layers nonlin (relu) except the last

        if len(nouts) > 1: self.layers.append(Layer(nin=nin, nout=nouts[0], nonlin=True))
        else: self.layers.append(Layer(nin=nin, nout=nouts[0], nonlin=False))

        for i, width in enumerate(nouts[1:-1], start=1):
            self.layers.append(Layer(nin=nouts[i-1], nout=width, nonlin=True, seed=42 + i))

        if len(nouts) > 1: self.layers.append(Layer(nin=nouts[-2], nout=nouts[-1], nonlin=False))

    def __call__(self, activations): # single inference
        for layer in self.layers:
            activations = layer(activations)
        return activations

    def forward_batch(self, activations):
        for layer in self.layers:
            activations = layer.forward_batch(activations)
        return activations

    def parameters(self):
        for layer in self.layers:
            yield from layer.parameters()
            

    def accumulate_gradients(self, gradients_forward):

        for layer in reversed(self.layers):
            gradients_forward = layer.accumulate_gradients(gradients_forward=gradients_forward)

    def accumulate_gradients_batch(self, gradients_forward):
        for layer in reversed(self.layers):
            gradients_forward = layer.accumulate_gradients_batch(gradients_forward)

    def __repr__(self):
            out = ""
            for k, layer in enumerate(self.layers):
                out += f"Layer {k}: {str(layer)}\n\n"
            return out
    
    def stats(self):
        header = f"{'Layer':<7}{'Inputs':>10}{'Neurons':>10}{'Params':>12}  {'Activation':<10}"
        line = "-" * len(header)

        print(line)
        print(header)
        print(line)

        total = 0
        for k, layer in enumerate(self.layers):
            params = layer.nout * (layer.nin + 1)   # weights + biases
            total += params
            act = "ReLU" if layer.nonlin else "Linear"
            print(f"{k:<7}{layer.nin:>10,}{layer.nout:>10,}{params:>12,}  {act:<10}")

        print(line)
        print(f"Layers: {len(self.layers)}")
        print(f"Total parameters: {total:,}")
        print(f"Memory (weights, float32): {total * 4 / 1e6:.2f} MB")
        print(line)


class SGD(Optimizer): # Standard Gradient Decent, dumb decent
    def __init__(self, model: Model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate

    def step_accumulate_gradients(self, x, y): # accumulates allowing for chunking

        z = self.model(x)
        loss, grad = z_loss_and_gradient(z=z, y=y)
        self.model.accumulate_gradients(gradients_forward=grad)

        return loss
    
    def step_learn(self):
        for p, g in self.model.parameters():
            g *= self.learning_rate              # in place, no temp
            p -= g   
        self.model.zero_grad()

    def step_accumulate_batch(self, x, y):  # x: (B, nin), y: (B, nout)
        z = self.model.forward_batch(x)
        loss, grad = z_loss_and_gradient_batch(z=z, y=y)
        self.model.accumulate_gradients_batch(gradients_forward=grad)
        return loss



def z_loss_and_gradient_batch(z, y):  # z, y: (B, nclasses), loss summed over the batch
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    sum_exp = exp_z.sum(axis=1, keepdims=True)
    grad = exp_z / sum_exp - y
    loss = (np.log(sum_exp) * y.sum(axis=1, keepdims=True)).sum() - (y * z_shifted).sum()
    return float(loss), grad

def z_loss_and_gradient(z, y):
    z_shifted = z - z.max()
    exp_z = np.exp(z_shifted)
    sum_exp = exp_z.sum()
    grad = exp_z / sum_exp - y
    loss = np.log(sum_exp) * y.sum() - y @ z_shifted   # -sum(y * log_softmax)
    return float(loss), grad

def z_loss(z, y): # only use when gradient is not needed, else prefer z_loss_and_gradient
    z_shifted = z - np.max(z)                      # for numerical stability
    log_probs = z_shifted - np.log(np.sum(np.exp(z_shifted)))   # log-softmax
    return -np.sum(y * log_probs)
    