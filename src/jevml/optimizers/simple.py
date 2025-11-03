import torch
from d2l import torch as d2l


class SimpleSGD(d2l.HyperParameters):
    """Simple minibatch stochastic gradient descent."""

    def __init__(self, params, lr):
        super().__init__()
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()