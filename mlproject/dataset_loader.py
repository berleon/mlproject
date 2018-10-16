import os
import copy
import torchvision
import torch
from mlproject.datasets import ClutteredMNIST


class DatasetLoader:
    def __init__(self,
                 train_set=None, train_generator=None,
                 test_set=None, test_generator=None,
                 validation_set=None, validation_generator=None):
        self._train_set = train_set
        self._train_generator = train_generator
        self._test_set = test_set
        self._test_generator = test_generator
        self._validation_set = validation_set
        self._validation_generator = validation_generator

    def train_set(self):
        return self._train_set

    def train_generator(self):
        return self._train_generator

    def test_set(self):
        return self._test_set

    def test_generator(self):
        return self._train_generator

    def validation_set(self):
        return self._validation_set

    def validation_generator(self):
        return self._validation_generator

    def has_train_set(self):
        return self.train_set() is not None

    def has_test_set(self):
        return self.test_set() is not None

    def has_validation_set(self):
        return self.validation_set() is not None

    # TODO: implement state_dict for subclasses. Really needed?
    def state_dict(self):
        return {}


def default_data_dir(maybe_data_dir=None):
    if maybe_data_dir is not None:
        return maybe_data_dir
    elif "DATA_DIR" in os.environ:
        return os.environ['DATA_DIR']
    else:
        raise ValueError("Can not figure out data_dir. "
                         "Please set the DATA_DIR enviroment variable.")


class TorchvisionDatasetLoader(DatasetLoader):
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


class CIFARDatasetLoader(TorchvisionDatasetLoader):
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


class FashionMNISTDatasetLoader(TorchvisionDatasetLoader):
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


class MNISTDatasetLoader(TorchvisionDatasetLoader):
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


class ClutteredMNISTDatasetLoader(TorchvisionDatasetLoader):
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
