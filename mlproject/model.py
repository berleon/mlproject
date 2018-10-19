from torch import nn


class Model(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self._device_args = [device]
        self._device_kwargs = {}

    def set_device_from_model(self, model):
        """Specific name of the model."""
        self._device_args = []
        self._device_kwargs = {
            'device': list(model.parameters())[0].device,
            'dtype': list(model.parameters())[0].type(),
        }

    def name(self):
        """Specific name of the model."""
        return self.__class__.__name__

    def train_batch(self, batch) -> {}:
        """Train the model with the given batch.
        `self.metrics` decides which outputs are logged."""
        raise NotImplementedError()

    def test_batch(self, batch) -> {}:
        """Test the model with the given batch. `self.benchmark_metric` decides which loss is
        used to compare two different models."""
        raise NotImplementedError()

    def on_train_end(self):
        """Callback that is called at the end of training."""
        pass

    def on_train_begin(self):
        """Callback that is called at the beginning of training."""
        pass

    def on_epoch_end(self, epoch):
        """Callback that is called at the end of an epoch."""
        pass

    def on_epoch_begin(self, epoch):
        """Callback that is called at the beginnnng of an epoch."""
        pass

    def benchmark_metric(self):
        """Metric to benchmark the model."""
        return 'loss'

    def metrics(self):
        """List of metrics that will be logged."""
        return ['loss']

    def to(self, *args, **kwargs):
        self._device_args = args
        self._device_kwargs = kwargs
        super().to(*args, **kwargs)

    def to_device(self, batch):
        """Moves batch to the same device of the model."""
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
