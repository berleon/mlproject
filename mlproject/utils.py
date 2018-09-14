import torch
import numpy as np
import io
import os
from PIL import Image


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
