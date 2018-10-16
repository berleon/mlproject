import torch
import numpy as np
import io
import os
from PIL import Image
import copy


def get_optimizer(config, parameters):
    """
    Loads optimizer from config dict.
    `name`: optimizer name, e.g. Adam, SGD, ...
    """
    opt_cls = getattr(torch.optim, config['name'])
    kwargs = copy.copy(config)
    del kwargs['name']
    return opt_cls(params=parameters, **kwargs)


def to_numpy(x):
    return x.to(torch.device("cpu")).detach().numpy()


def to_channels_last(x):
    assert x.ndim == 3
    return np.rollaxis(x, 0, 3)


def to_channels_first(x):
    assert x.ndim == 3
    return np.rollaxis(x, -1, 0)


def savefig_as_bytes(fig=None, **kwargs):
    buf = io.BytesIO()
    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt
    fig.savefig(buf, format='png', **kwargs)
    buf.seek(0)
    return buf


def savefig_as_np(fig=None, **kwargs):
    buf = savefig_as_bytes(fig, **kwargs)
    im = Image.open(buf)
    np_arr = np.array(im)
    return np_arr[:, :, :3]


def print_environment_vars():
    for envname in ['DATA_DIR', 'TENSORBOARD_DIR']:
        print("{}: {}".format(envname, os.environ.get(envname, '')))


def int_as_tuple(x):
    if type(x) == int:
        return (x, x)
    else:
        return x


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get


class GeneratorWithLenght:
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen
