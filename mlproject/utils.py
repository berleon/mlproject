import torch
import numpy as np
import io
import os
from PIL import Image
import copy

import pickle
import struct


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


# https://stackoverflow.com/questions/5580201/seek-into-a-file-full-of-pickled-objects


class IndexedPickleReader:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.f = open(self.filename, 'r+b')
        loc, = struct.unpack('L', self.f.read(8))
        self.f.seek(loc)
        self.indicies = pickle.load(self.f)
        return self

    def __exit__(self, *args):
        self.f.close()

    def __getitem__(self, idx):
        loc = self.indicies[idx]
        self.f.seek(loc)
        return pickle.load(self.f)


class IndexedPickleWriter:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.f = open(self.filename, 'w+b')
        self.f.write(struct.pack('L', 0))
        self.indicies = []
        return self

    def __exit__(self, *args):
        loc = self.f.tell()
        pickle.dump(self.indicies, self.f)
        self.f.seek(0)
        self.f.write(struct.pack('L', loc))
        self.f.close()

    def dump(self, obj):
        assert hasattr(self, 'f')
        self.indicies.append(self.f.tell())
        pickle.dump(obj, self.f)
