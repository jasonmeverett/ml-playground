import torch
import random
from d2l import torch as d2l

class SyntheticRegressionData(d2l.DataModule):
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        
        # We model observation noise on the output. This is an nx1, where each
        # row is the observational noise of the specific label.
        obs_noise = torch.randn(n, 1) * noise
        self.y = torch.mm(self.X, w.reshape((-1, 1))) + b + obs_noise

    def get_tensorloader(self, tensors, train = True, indices = slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

    def get_dataloader(self, train = True):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)
