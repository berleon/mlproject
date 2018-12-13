import torch
from mlproject.log import LogLevel, DevNullSummaryWriter


class Trainer:
    """
    This call holds the model (the pytorch layers and weights) but
    also knows how to train the model for a single given batch.
    For simple cases, such as classification you might be able to
    use the ``SimpleTrainer`` class.  When subclassing, you have to
    implement the ``train_batch`` and `test_batch` methods.

    The ``MLProject`` class will add a log writer to the model (see
    ``mlproject.log``). And thus it can log scalars and images.
    """

    def __init__(self, model, writer=None, device='cpu'):
        super().__init__()
        self.model = model
        self._device_args = [device]
        self._device_kwargs = {}
        self.log = LogLevel.SCALARS
        self.writer = writer or DevNullSummaryWriter()

    def set_writer(self, writer):
        self.writer = writer

    def set_device_from_model(self, model):
        """Specific name of the model."""
        self._device_args = []
        self._device_kwargs = {
            'device': list(model.parameters())[0].device,
            'dtype': list(model.parameters())[0].type(),
        }

    def train_batch(self, batch) -> {}:
        """
        Train the model with the given batch and return some useful statistics.
        The optimization step should happen inside here. `self.metrics` decides
        which outputs are logged.  You can inspect the log level with
        `self.log`. The number of log iterations can be configured using the
        `log_iteration_scalars` and `log_iteration_all` config
        entries. Use the `self.log` member to skip costly computations.

        A possible implementation could look like this:

        ```
        def train_batch(self, batch) -> {}:
            self.opt.zero_grad()
            imgs, labels = self.to_device(batch)
            logits = self(imgs)
            loss = - nll(logits, labels)
            loss.backward()

            self.opt.step()
            if self.log.SCALARS:
                return {'loss': loss.item()}
            elif self.log.ALL:
                return {
                    'loss': loss.item(),
                    'images': self.takes_long_time_to_generate(imgs)
                }
        ```
        """
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

    def minimize_benchmark_metric(self):
        return True

    def metrics(self):
        """List of metrics that will be logged."""
        return ['loss']

    @property
    def device(self):
        return torch.device(*self._device_args, **self._device_kwargs)

    def to(self, *args, **kwargs):
        self._device_args = args
        self._device_kwargs = kwargs
        self.model.to(*args, **kwargs)

    def to_device(self, batch):
        """Moves batch to the same device of the model."""
        if type(batch) in (list, tuple):
            return (x.to(*self._device_args, **self._device_kwargs) for x in batch)
        else:
            return batch.to(*self._device_args, **self._device_kwargs)

    def state_dict(self):
        raise NotImplementedError()


class SimpleTrainer(Trainer):
    """
    For simple models, you can use this class. Given a model,
    optimizer and a loss layer, this class puts them together in
    a straight way.
    """

    def __init__(self, model, optimizer: torch.optim.Optimizer, loss, device='cpu',
                 optimizer_state=None):
        super().__init__(device)
        self.model = model
        self.optimizer = optimizer
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.loss = loss

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

    def state_dict(self):
        return {'optimizer_state': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
