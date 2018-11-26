# MLProject

This is project aims to provide a init for a machine learning project with [sacred](https://sacred.readthedocs.io/en/latest/),
pytorch, and tensorboard.


See the [test](test/test_cifar.py) for an example.

This framework tries to force you to separate the model, dataset, training and
evaluation from each other. You might want train a model on CIFAR and MNIST or you want to see if SVM or a MLP is best on MNIST. However, models always depend on datasets, e.g on the input sizes. This repository doesn't try to fix this.


## Model

Your model class has to implement the following methods:

```python
class MyModel(mlproject.Model):
    def train_batch(self, batch) -> {}:
        """Train the model with the given batch.
        `self.metrics` decides which outputs are logged."""
        raise NotImplementedError()

    def test_batch(self, batch) -> {}:
        """Test the model with the given batch. `self.benchmark_metric` decides which loss is
        used to compare two different models."""
        raise NotImplementedError()
```

Compared to a standard PyTorch `nn.Module` class the `mlproject.Model` owns its
optimizers.

## Data

The dataset are loaded with a DatasetFactory. The class provides access to
the train/test/val datasets and all the dataset iterators.

A few dataset loaders are already implemented for:
* MNIST
* ClutteredMNIST
* FashionMNIST
* CIFAR10

You can implement a dataset factory on your own. Just see the API of the
[DatasetFactory](mlproject/data.py) class.


## MLProject

The model and the data have to interact in a specific way. This is what the
`MLProject` class takes care of. In the simplest case it is take the data, put
the data in the model and minimize the loss for n epochs.
However, training loops can become complicated. You can always overwrite the
`train_epoch`  method and adapt it to your needs.

The MLProject also has to know how to load the model and the data. Therefore,
you have to subclass it and implement the `get_dataset_factory` and `get_model`
functions.

For example:

```
class CifarProject(MLProject):
    @staticmethod
    def get_dataset_factory(config):
        return CIFARDatasetLoader(
            config['batch_size'],
            train_transform=Compose([
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            test_transform=Compose([
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    @staticmethod
    def get_model(config):
        net = ResNet18()
        if torch.cuda.is_available():
            net.to("cuda:0")
        opt = torch.optim.Adam(net.parameters())
        return ProxyModel(net, opt, loss=nn.CrossEntropyLoss(), name='test_cifar')
```

## Sacred

Sacred is used to keep track of the experiments.
There are a few sacred configurations that will be used by the `MLProject` class.
You can overwrite all methods in `MLProject` and use your own
configuration names.

Required:
* `n_epochs`: Train the model for this number of epochs. Can be
  omitted if `n_global_iterations` is given.
* `n_global_iterations`: Train the model for at most this number of
  iterations. If given, it has precedence over `n_epochs`. Default: `0`
* `model_dir`: Saves the model to this directory.

Optional:
* `device`: The device to run your model on, e.g. `cuda:0`. Default: `cpu`
* `tensorboard_dir`: Save the tensorboard run in this directory. If `None`,
  tensorboard is not used. Default: `None`
* `log_iteration_scalars`: number of iterations when to log scalars. Default: `1`
* `log_iteration_all`: number of iterations when to log everything. Use `epoch`
  to log it at the end of every epoch. Default: `epoch`
* `save_iterations`: number of iterations when save the model. Use `epoch` to
  save it at the end of every epoch. Default: `epoch`
* `prefix`: use this prefix for tensorboard logs and models saves

## Makefile

I find it handy to have a Makefile that then run the experiments.
Something like this:

```
train:
    ./main.py with tensorboard_dir=$(TENSORBOARD_DIR)
```
