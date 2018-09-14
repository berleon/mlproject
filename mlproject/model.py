from torch import nn


class Model(nn.Module):
    def name(self):
        raise NotImplementedError()

    def train_batch(self, batch) -> {}:
        raise NotImplementedError()

    def test_batch(self, batch) -> {}:
        raise NotImplementedError()

    def on_train_end(self):
        pass

    def on_train_begin(self):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def benchmark_metric(self):
        return 'loss'


class TorchModel(Model):
    def __init__(self, model, optimizer, loss, name):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.optimizer = optimizer
        self.loss = loss
        self._name = name

    def name(self):
        return self._name

    def to_device(self, batch):
        input, labels = batch
        input = input.to(self.device)
        labels = labels.to(self.device)
        return input, labels

    def train_batch(self, batch):
        input, labels = self.to_device(batch)
        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.loss(output, labels)
        loss.backward()
        self.optimizer.step()
        return {'loss': loss}

    def test_batch(self, batch):
        input, labels = self.to_device(batch)
        output = self.model(input)
        return {'loss': self.loss(output, labels)}
