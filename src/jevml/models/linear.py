import torch
from d2l import torch as d2l

from jevml.optimizers.simple import SimpleSGD

class SimpleLinear(d2l.Module):
    """Linear Regression Model
    """
    
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, [num_inputs, 1], requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return torch.mm(X, self.w) + self.b

    def loss(self, y_hat, y):
        l = (y_hat - y) ** 2 / 2
        return l.mean()

    def configure_optimizers(self):
        return SimpleSGD([self.w, self.b], self.lr)
