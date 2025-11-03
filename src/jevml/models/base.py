import torch
from d2l import torch as d2l

from jevml.optimizers.simple import SimpleSGD


class Classifier(d2l.Module):
    def validation_step(self, batch):
        y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


    def accuracy(self, y_hat, y, averaged=True):
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        preds = y_hat.argmax(axis=1).type(y.dtype)
        compare = (preds == y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare