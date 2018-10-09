import numpy as np
from mlproject.dataset_loader import CIFARDatasetLoader, MNISTDatasetLoader, \
    ClutteredMNISTDatasetLoader


def test_cifar_data_loader():
    cifar = CIFARDatasetLoader()
    assert cifar.has_train_set()
    assert cifar.has_test_set()
    assert not cifar.has_validation_set()

    img, label = cifar.train_set()[0]
    assert np.array(img).shape == (32, 32, 3)


def test_mnist_data_loader():
    mnist = MNISTDatasetLoader()
    assert mnist.has_train_set()
    assert mnist.has_test_set()
    assert not mnist.has_validation_set()

    img, label = mnist.train_set()[0]
    assert np.array(img).shape == (28, 28)


def test_cluttered_mnist_data_loader():
    cluttered_mnist = ClutteredMNISTDatasetLoader(shape=(100, 100), n_samples=100)
    assert cluttered_mnist.has_train_set()
    assert cluttered_mnist.has_test_set()
    assert not cluttered_mnist.has_validation_set()

    img, label = cluttered_mnist.train_set()[0]
    assert np.array(img).shape == (100, 100)
