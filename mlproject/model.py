from torch import nn


class Model(nn.Module):
    def __init__(self, device='cpu'):
        self._device_args = [device]
        self._device_kwargs = {}

    def set_device_from_model(self, model):
        self._device_args = []
        self._device_kwargs = {
            'device': list(model.parameters())[0].device,
            'dtype': list(model.parameters())[0].type(),
        }

    def name(self):
        return self.__class__.__name__

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

    def to(self, *args, **kwargs):
        self._device_args = args
        self._device_kwargs = kwargs
        super().to(*args, **kwargs)

    def to_device(self, batch):
        if type(batch) in (list, tuple):
            return (x.to(*self._device_args, **self._device_kwargs) for x in batch)
        else:
            return batch.to(*self._device_args, **self._device_kwargs)


class ClassificationModel(Model):
    def __init__(self, model, optimizer, loss, name):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self._name = name
        self.set_device_from_model(self.model)

    def name(self):
        return self._name

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
