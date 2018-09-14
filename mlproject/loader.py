import os
import torchvision
import torch


class DatasetLoader:
    def train_set(self):
        raise NotImplemented()

    def train_generator(self):
        raise NotImplemented()

    def test_set(self):
        raise NotImplemented()

    def test_generator(self):
        raise NotImplemented()

    def validation_set(self):
        raise NotImplemented()

    def validation_generator(self):
        raise NotImplemented()

    def has_train_set(self):
        try:
            self.train_set()
            return True
        except NotImplemented:
            return False

    def has_validation_set(self):
        try:
            self.validation_set()
            return True
        except NotImplemented:
            return False


def default_data_dir(maybe_data_dir=None):
    if maybe_data_dir is not None:
        return maybe_data_dir
    elif "DATA_DIR" in os.environ:
        return os.environ['DATA_DIR']
    else:
        raise ValueError("Can not figure out data_dir. "
                         "Please set the DATA_DIR enviroment variable.")


class CIFARDatasetLoader(DatasetLoader):
    def __init__(self, batch_size=50, train_transform=None, test_transform=None, data_dir=None):
        self.batch_size = batch_size
        self.data_dir = default_data_dir(data_dir)

        self._trainset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=train_transform)
        self._testset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=test_transform)

        self._trainloader = torch.utils.data.DataLoader(
            self._trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        self._testloader = torch.utils.data.DataLoader(
            self._testset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def train_set(self):
        return self._trainset

    def train_generator(self):
        return self._trainloader

    def test_set(self):
        return self._testset

    def test_generator(self):
        return self._testloader
