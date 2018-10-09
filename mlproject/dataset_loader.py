import os
import torchvision
import torch
from mlproject.datasets import ClutteredMNIST


class DatasetLoader:
    def train_set(self):
        pass

    def train_generator(self):
        pass

    def test_set(self):
        pass

    def test_generator(self):
        pass

    def validation_set(self):
        pass

    def validation_generator(self):
        pass

    def has_train_set(self):
        return self.train_set() is not None

    def has_test_set(self):
        return self.test_set() is not None

    def has_validation_set(self):
        return self.validation_set() is not None


def default_data_dir(maybe_data_dir=None):
    if maybe_data_dir is not None:
        return maybe_data_dir
    elif "DATA_DIR" in os.environ:
        return os.environ['DATA_DIR']
    else:
        raise ValueError("Can not figure out data_dir. "
                         "Please set the DATA_DIR enviroment variable.")


class TorchvisionDatasetLoader(DatasetLoader):
    def __init__(self, trainset=None, testset=None, valset=None, batch_size=50,
                 n_workers=0):
        self.batch_size = batch_size

        self._trainset = trainset
        self._testset = testset
        self._valset = valset

        if self._trainset is not None:
            self._trainloader = torch.utils.data.DataLoader(
                self._trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        else:
            self._trainloader = None

        if self._testset is not None:
            self._testloader = torch.utils.data.DataLoader(
                self._testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        else:
            self._testloader = None

        if self._valset is not None:
            self._valloader = torch.utils.data.DataLoader(
                self._testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        else:
            self._valloader = None

    def train_set(self):
        return self._trainset

    def train_generator(self):
        return self._trainloader

    def test_set(self):
        return self._testset

    def test_generator(self):
        return self._testloader

    def validation_set(self):
        return self._valloader

    def validation_generator(self):
        return self._valloader


class CIFARDatasetLoader(TorchvisionDatasetLoader):
    def __init__(self, batch_size=50, train_transform=None, test_transform=None, data_dir=None,
                 n_workers=0):
        self.data_dir = default_data_dir(data_dir)
        trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                               download=True, transform=test_transform)
        super().__init__(trainset, testset, batch_size=batch_size, n_workers=n_workers)


class MNISTDatasetLoader(TorchvisionDatasetLoader):
    def __init__(self, batch_size=50, train_transform=None, test_transform=None, data_dir=None,
                 n_workers=0):
        self.data_dir = default_data_dir(data_dir)
        trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True,
                                              download=True, transform=train_transform)
        testset = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                             download=True, transform=test_transform)
        super().__init__(trainset, testset, batch_size=batch_size, n_workers=n_workers)


class ClutteredMNISTDatasetLoader(TorchvisionDatasetLoader):
    def __init__(self, batch_size=50, train_transform=None, test_transform=None, data_dir=None,
                 shape=(100, 100), n_clutters=6, clutter_size=8, n_samples=60000,
                 n_workers=0):
        self.data_dir = default_data_dir(data_dir)
        trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True,
                                              download=True, transform=train_transform)
        testset = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                             download=True, transform=test_transform)
        cluttered_train = ClutteredMNIST(trainset, shape, n_clutters, clutter_size, n_samples)
        cluttered_test = ClutteredMNIST(testset, shape, n_clutters, clutter_size, n_samples)
        super().__init__(cluttered_train, cluttered_test,
                         batch_size=batch_size, n_workers=n_workers)
