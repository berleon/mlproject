import os
import copy
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from mlproject.datasets import ClutteredMNIST


class DatasetFactory:
    """
    The DatasetFactory provides access to the train/test/val datasets and all the dataset
    iterators.

    The `Dataset`'s returned by `train_set`, `test_set`, and `validation_set`
    should be persistent, e.g.  an index should always return the same data and
    label.

    The `DataLoaders` returned by *_loader can of course return the data
    augmented and in any order.
    """
    def __init__(self,
                 train_set=None, train_loader=None,
                 test_set=None, test_loader=None,
                 validation_set=None, validation_loader=None):
        self._train_set = train_set
        self._train_loader = train_loader
        self._test_set = test_set
        self._test_loader = test_loader
        self._validation_set = validation_set
        self._validation_loader = validation_loader

    def train_set(self) -> Dataset:
        """Return the train set."""
        return self._train_set

    def train_loader(self) -> DataLoader:
        """Return the DataLoader associated with the train set."""
        return self._train_loader

    def test_set(self) -> Dataset:
        """Return the test set."""
        return self._test_set

    def test_loader(self) -> DataLoader:
        """Return the DataLoader associated with the test set."""
        return self._train_loader

    def validation_set(self) -> Dataset:
        """Return the validation set."""
        return self._validation_set

    def validation_loader(self) -> DataLoader:
        """Return the DataLoader associated with the validation set."""
        return self._validation_loader

    def has_train_set(self) -> bool:
        return self.train_set() is not None

    def has_test_set(self) -> bool:
        return self.test_set() is not None

    def has_validation_set(self) -> bool:
        return self.validation_set() is not None


def default_data_dir(maybe_data_dir=None):
    if maybe_data_dir is not None:
        return maybe_data_dir
    elif "DATA_DIR" in os.environ:
        return os.environ['DATA_DIR']
    else:
        raise ValueError("Can not figure out data_dir. "
                         "Please set the DATA_DIR enviroment variable.")


class TorchvisionDatasetFactory(DatasetFactory):
    # TODO: Extract more code from subclass
    def __init__(self, train_set=None, test_set=None, validation_set=None,
                 data_loader_kwargs={},
                 data_loader_train_kwargs={},
                 data_loader_test_kwargs={}):
        train_kwargs = copy.copy(data_loader_kwargs)
        train_kwargs.update(data_loader_train_kwargs)
        test_kwargs = copy.copy(data_loader_kwargs)
        test_kwargs.update(data_loader_train_kwargs)

        if train_set is not None:
            trainloader = torch.utils.data.DataLoader(train_set, **train_kwargs)
        else:
            self._trainloader = None

        if test_set is not None:
            testloader = torch.utils.data.DataLoader(test_set, **test_kwargs)
        else:
            testloader = None

        if validation_set is not None:
            valloader = torch.utils.data.DataLoader(validation_set, **test_kwargs)
        else:
            valloader = None

        super().__init__(
            train_set, trainloader,
            test_set, testloader,
            validation_set, valloader,
        )


class CIFARDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size=50, train_transform=None, test_transform=None, data_dir=None,
                 num_workers=0):
        self.data_dir = default_data_dir(data_dir)
        trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                               download=True, transform=test_transform)
        super().__init__(
            trainset, testset,
            data_loader_kwargs={
                'num_workers': num_workers,
                'batch_size': batch_size,
            },
            data_loader_train_kwargs={'shuffle': True}
        )


class FashionMNISTDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size=1, train_transform=None, test_transform=None, data_dir=None,
                 num_workers=0, collate_fn=None, pin_memory=None, drop_last=None,
                 data_loader_train_kwargs=None,
                 data_loader_test_kwargs=None,
                 ):
        def update_kwargs(kwargs):
            kwargs = copy.copy(kwargs or {})
            if 'num_workers' not in kwargs:
                kwargs['num_workers'] = num_workers
            if 'collate_fn' not in kwargs and collate_fn is not None:
                kwargs['collate_fn'] = collate_fn
            if 'pin_memory' not in kwargs and pin_memory is not None:
                kwargs['pin_memory'] = pin_memory
            if 'drop_last' not in kwargs and drop_last is not None:
                kwargs['drop_last'] = drop_last
            if 'batch_size' not in kwargs:
                kwargs['batch_size'] = batch_size
            return kwargs

        self.data_dir = default_data_dir(data_dir)
        trainset = torchvision.datasets.FashionMNIST(root=self.data_dir, train=True,
                                                     download=True, transform=train_transform)
        testset = torchvision.datasets.FashionMNIST(root=self.data_dir, train=False,
                                                    download=True, transform=test_transform)
        super().__init__(
            trainset, testset,
            data_loader_train_kwargs=update_kwargs(data_loader_train_kwargs),
            data_loader_test_kwargs=update_kwargs(data_loader_test_kwargs)
        )


class MNISTDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size=50, train_transform=None, test_transform=None, data_dir=None,
                 num_workers=0):
        self.data_dir = default_data_dir(data_dir)
        trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True,
                                              download=True, transform=train_transform)
        testset = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                             download=True, transform=test_transform)
        super().__init__(
            trainset, testset,
            data_loader_kwargs={
                'num_workers': num_workers,
                'batch_size': batch_size,
            },
            data_loader_train_kwargs={'shuffle': True}
        )


class ClutteredMNISTDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size=50, train_transform=None, test_transform=None, data_dir=None,
                 shape=(100, 100), n_clutters=6, clutter_size=8, n_samples=60000,
                 num_workers=0):
        self.data_dir = default_data_dir(data_dir)
        trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True)
        testset = torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True)
        cluttered_train = ClutteredMNIST(trainset, shape, n_clutters,
                                         clutter_size, n_samples,
                                         transform=train_transform)
        cluttered_test = ClutteredMNIST(testset, shape, n_clutters,
                                        clutter_size, n_samples,
                                        transform=test_transform)
        super().__init__(
            cluttered_train, cluttered_test,
            data_loader_kwargs={
                'num_workers': num_workers,
                'batch_size': batch_size,
            },
            data_loader_train_kwargs={'shuffle': True}
        )
