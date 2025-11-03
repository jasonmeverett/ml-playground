import torch
from d2l import torch as d2l

from jevml.models.base import Classifier
from jevml.utils import softmax, cross_entropy

class SoftmaxRegression(Classifier):
    """Softmax Regression Model
    """
    
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, [num_inputs, num_outputs], requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]

    def forward(self, X):
        X = X.reshape((-1, self.W.shape[0]))
        return softmax(torch.mm(X, self.W) + self.b)

    def loss(self, y_hat, y):
        return cross_entropy(y_hat, y)