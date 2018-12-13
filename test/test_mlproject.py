from mlproject.mlproject import MLProject
from sacred import Experiment
from mlproject.data import CIFARDatasetFactory
from mlproject.trainer import SimpleTrainer
from mlproject.log import LogLevel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip


def dataloader(**config):
    return CIFARDatasetFactory(
        config['batch_size'],
        train_transform=Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        test_transform=Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3*32*32, 10)

    def forward(self, x):
        bs, ch, h, w = x.shape
        return self.linear(x.view(bs, ch*h*w))


class MyTrainer(SimpleTrainer):
    def __init__(self, model):
        super().__init__(model, torch.optim.SGD(model.parameters(), lr=0.1), nn.CrossEntropyLoss())


class MyProject(MLProject):
    @staticmethod
    def get_dataset_factory(config):
        return CIFARDatasetFactory(
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
        return LinearModel()

    @staticmethod
    def get_trainer(model, config):
        return MyTrainer(model)


def test_mlproject_global_iteration(tmpdir):
    N_GLOBAL_ITERATIONS = 3
    ex = Experiment()
    ex.add_config({
        'name': 'test',
        'batch_size': 5,
        'n_global_iterations':  N_GLOBAL_ITERATIONS,
        'tensorboard_dir': None,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'model_dir': str(tmpdir.join('models')),
    })

    @ex.automain
    def main(_run):
        proj = MyProject.from_run(_run)
        proj.train()
        assert proj.global_step == N_GLOBAL_ITERATIONS

    ex.run()


def test_mlproject_log_iterations(tmpdir):
    N_GLOBAL_ITERATIONS = 10
    ex = Experiment()
    ex.add_config({
        'name': 'test',
        'batch_size': 5,
        'n_global_iterations':  N_GLOBAL_ITERATIONS,
        'log_iteration_scalars': 2,
        'log_iteration_all': 5,
        'tensorboard_dir': None,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'model_dir': str(tmpdir.join('models')),
    })

    @ex.automain
    def main(_run):
        class MyModel(SimpleTrainer):
            def __init__(self, model):
                super().__init__(model, torch.optim.SGD(model.parameters(), lr=0.1),
                                 nn.CrossEntropyLoss())

            def train_batch(self, batch):
                if proj.epoch_step % proj.config['log_iteration_scalars'] == 0:
                    self.log = LogLevel.SCALARS
                if proj.epoch_step % proj.config['log_iteration_all'] == 0:
                    self.log = LogLevel.ALL
                super().train_batch(batch)

        class TestProject(MyProject):
            @staticmethod
            def get_model(config):
                return LinearModel()

            @staticmethod
            def get_trainer(model, config):
                return MyTrainer(model)

        proj = MyProject.from_run(_run)
        proj.train()
        assert proj.global_step == N_GLOBAL_ITERATIONS

    ex.run()
