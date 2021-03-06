import numpy as np
from mlproject.data import CIFARDatasetFactory, MNISTDatasetFactory, \
    ClutteredMNISTDatasetFactory, FashionMNISTDatasetFactory
from torchvision.transforms import ToTensor


def test_cifar_data_factory():
    cifar = CIFARDatasetFactory()
    assert cifar.has_train_set()
    assert cifar.has_test_set()
    assert not cifar.has_validation_set()

    img, label = cifar.train_set()[0]
    assert np.array(img).shape == (32, 32, 3)


def test_mnist_data_factory():
    mnist = MNISTDatasetFactory()
    assert mnist.has_train_set()
    assert mnist.has_test_set()
    assert not mnist.has_validation_set()

    img, label = mnist.train_set()[0]
    assert np.array(img).shape == (28, 28)


def test_cluttered_mnist_data_factory():
    cluttered_mnist = ClutteredMNISTDatasetFactory(
        shape=(100, 100),
        n_samples=100,
        batch_size=33,
        train_transform=ToTensor(),
        test_transform=ToTensor(),
    )
    assert cluttered_mnist.has_train_set()
    assert cluttered_mnist.has_test_set()
    assert not cluttered_mnist.has_validation_set()

    img, label = cluttered_mnist.train_set()[0]
    assert np.array(img).shape == (1, 100, 100)

    imgs, labels = next(iter(cluttered_mnist.train_loader()))
    assert np.array(imgs).shape == (33, 1, 100, 100)
    assert np.array(labels).shape == (33,)


def test_fashionmnist_data_factory():
    fashin_mnist = FashionMNISTDatasetFactory(
        batch_size=33,
        train_transform=ToTensor(),
        test_transform=ToTensor(),
    )
    assert fashin_mnist.has_train_set()
    assert fashin_mnist.has_test_set()
    assert not fashin_mnist.has_validation_set()

    img, label = fashin_mnist.train_set()[0]
    assert np.array(img).shape == (1, 28, 28)

    imgs, labels = next(iter(fashin_mnist.train_loader()))
    assert np.array(imgs).shape == (33, 1, 28, 28)
    assert np.array(labels).shape == (33,)
