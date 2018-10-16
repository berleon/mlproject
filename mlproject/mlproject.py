import os
from collections import OrderedDict
import sacred
from tqdm import tqdm
import torch
import random
from tensorboardX import SummaryWriter

from mlproject.log import get_tensorboard_dir, DevNullSummaryWriter, set_global_writer
from mlproject.utils import to_numpy, print_environment_vars


def get_model_dir(config, model_identifier):
    if "model_dir" in config:
        model_dir = config['model_dir']
    elif 'MODEL_DIR' in os.environ:
        model_dir = os.environ['MODEL_DIR']
    else:
        raise Exception("Cannot figure out model dir.")
    return os.path.join(model_dir, model_identifier)


class MLProject:
    def __init__(self,
                 _id,
                 config,
                 global_step=0,
                 epoch=0,
                 epoch_step=0,
                 best_score=0,
                 model_save_dir=None,
                 tensorboard_log_dir=None,
                 model_state=None,
                 _run=None,
                 ):
        self._id = _id or random.randint(int(1e10), int(1e10) + int(1e8))
        self.config = config
        self._run = _run
        self.dataset_loader = self.get_dataset_loader(self.config)
        self.model = self.get_model(self.config)
        if model_state:
            self.model.load_state_dict(model_state)

        if 'device' in self.config:
            device_name = self.config['device']
        else:
            if torch.cuda.is_available():
                device_name = 'cuda:0'
            else:
                device_name = 'cpu'

        self.device = torch.device(device_name)
        self.model.to(self.device)

        self.global_step = global_step
        self.epoch = epoch
        self.epoch_step = epoch_step
        self.best_score = None
        if tensorboard_log_dir is None and self.config.get('tensorboard', False):
            self.tensorboard_log_dir = get_tensorboard_dir(
                str(self._id) + '_' + self.model.name())
        else:
            self.tensorboard_log_dir = tensorboard_log_dir
        if self.tensorboard_log_dir is None:
            self.writer = DevNullSummaryWriter()
        else:
            self.writer = SummaryWriter(self.tensorboard_log_dir)
        if model_save_dir is None:
            self.model_save_dir = get_model_dir(self.config,
                                                str(self._id) + '_' + self.model.name())
            os.makedirs(self.model_save_dir)
        else:
            self.model_save_dir = model_save_dir
        set_global_writer(self.writer)

    @classmethod
    def from_run(cls, _run: sacred.run.Run):
        sacred.commands.print_config(_run)
        print_environment_vars()
        cfg = _run.config
        return cls(_run._id, cfg, _run=_run)

    @staticmethod
    def get_model(config):
        raise NotImplementedError()

    @staticmethod
    def get_dataset_loader(config):
        raise NotImplementedError()

    def _is_better(self, score):
        if self.best_score is None:
            return True
        elif self.model.benchmark_metric() == 'accuracy':
            return self.best_score < score
        elif self.model.benchmark_metric() == 'nll':
            return self.best_score > score
        else:
            raise Exception()

    def test(self):
        # TODO: seperate validation and test
        self.model.eval()
        test_losses = OrderedDict()
        n_test_samples = len(self.dataset_loader.test_set())
        for batch in self.dataset_loader.test_generator():
            losses = self.model.test_batch(batch)
            for name, value in losses.items():
                if name not in test_losses:
                    test_losses[name] = 0
                test_losses[name] += float(to_numpy(value) / n_test_samples)

        loss_info = ", ".join(["{}: {:.4f}".format(name, float(loss))
                               for name, loss in sorted(test_losses.items())])
        self.writer.add_scalars('test', test_losses, self.global_step)
        print("[TEST] " + loss_info)
        return test_losses[self.model.benchmark_metric()]

    def train(self):
        self.model.on_train_begin()
        self.model.train()
        best_model_fname = None
        for epoch_idx in range(self.config['n_train_epochs']):
            self.train_epoch()
            if self.dataset_loader.has_test_set():
                score = self.test()
                if self._is_better(score):
                    best_model_fname = self.save()
                    self.best_score = score
            self.epoch += 1

        if best_model_fname is not None:
            if self._run is not None:
                self._run.add_artifact(best_model_fname)
        self.model.on_train_end()

    def train_epoch(self):
        progbar = tqdm(self.dataset_loader.train_generator())
        self.model.on_epoch_begin(self.epoch)
        self.epoch_step = 0
        for batch in progbar:
            losses = self.model.train_batch(batch)
            # TODO: fix display issue in tensorboard
            self.writer.add_scalars('training', losses, self.global_step)
            self.global_step += 1
            self.epoch_step += 1
        self.model.on_epoch_end(self.epoch)
        self.epoch += 1

    def state_dict(self):
        return {
            '_id': self._id,
            'config': self.config,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'epoch_step': self.epoch_step,
            'best_score': self.best_score,
            'model_save_dir': self.model_save_dir,
            'tensorboard_log_dir': self.tensorboard_log_dir,
            'model_state': self.model.state_dict(),
        }

    def save_filename(self):
        return os.path.join(
            self.model_save_dir,
            "{}_e{:05}_b{:05}.torch".format(self.model.name(), self.epoch, self.epoch_step))

    def save(self):
        torch.save(self.state_dict(), self.save_filename())
        return self.save_filename()

    @classmethod
    def load(cls, filename):
        return cls(**torch.load(filename))
