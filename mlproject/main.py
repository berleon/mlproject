#! /usr/bin/env python3
import matplotlib
matplotlib.use('agg')  # noqa
import torch

import sacred
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from sacred import Experiment

from attribution.db import add_mongodb, add_package_sources
from attribution.trainer import Trainer
from attribution.dataset_loader import get_dataset_loader

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


ex = Experiment("train")
ex.captured_out_filter = apply_backspaces_and_linefeeds
add_package_sources(ex)
add_mongodb(ex)

ex.add_config(**{
    'dataset': {
        'name': 'cifar',
        'batch_size': 50,
    },
    'model': {
        'name': 'baseline',
    },
    'tag': None,
    'epochs': 200,
})


@ex.automain
def main(_run: Run, dataset, model, epochs, tag):
    sacred.commands.print_config(_run)
    dataset_loader = get_dataset_loader(dataset)
    model = dataset_loader.load_model(**model)
    print(model)
    trainer = Trainer(dataset_loader, model, _run._id, tag, experiment=ex)
    trainer.test()
    return trainer.train(epochs)



def main(**config):
    ml_project(ex, config, data_loader, model_loader).train()
