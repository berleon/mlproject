import os
import copy
from collections import OrderedDict
import sacred
from tqdm import tqdm
import torch
import random
from tensorboardX import SummaryWriter

from mlproject.log import get_tensorboard_dir, DevNullSummaryWriter, set_global_writer
from mlproject.utils import to_numpy
from mlproject.data import DatasetFactory
from mlproject.model import Model, LogLevel


def get_model_dir(config, model_identifier):
    if "model_dir" in config:
        model_dir = config['model_dir']
    else:
        raise Exception("Cannot figure out model dir.")
    return os.path.join(model_dir, model_identifier)


class TrainingStop(Exception):
    pass

class MLProject:
    def __init__(self,
                 config,
                 _id=None,
                 global_step=0,
                 epoch=0,
                 epoch_step=0,
                 best_score=None,
                 model_save_dir=None,
                 tensorboard_run_dir=None,
                 model_state=None,
                 _run=None,
                 ):
        """
        The MLProject class is in carge of training the model and evaluating it.

        Args:
            _id (int):  Id of the current run
            config (dict): Config dictionary from sacred
            global_step (int): The global step of the model
            model_save_dir (str): The model directory to save it
            tensorboard_run_dir (str): Path to the tensorboard run directory of the model
            model_state (dict): model state_dict for loading weights
            _run (sacred.Run): current sacred run (optional)
        """
        # TODO: set default config
        self._id = _id or random.randint(int(1e10), int(1e10) + int(1e8))
        self.config = self.set_defaults(copy.deepcopy(config))
        self._run = _run
        self.dataset_factory = self.get_dataset_factory(self.config)
        self.model = self.get_model(self.config)
        self.prefix = self.config['prefix']
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
        self.best_score = best_score
        if tensorboard_run_dir is None and self.config.get('tensorboard_dir', None):
            self.tensorboard_run_dir = get_tensorboard_dir(
                str(self._id) + '_' + self.model.name())
        else:
            self.tensorboard_run_dir = tensorboard_run_dir
        if self.tensorboard_run_dir is None:
            self.writer = DevNullSummaryWriter()
        else:
            self.writer = SummaryWriter(self.tensorboard_run_dir)
        self.model.set_writer(self.writer)
        if model_save_dir is None:
            self.model_save_dir = get_model_dir(
                self.config, str(self._id) + '_' + self.model.name())
            os.makedirs(self.model_save_dir)
        else:
            self.model_save_dir = model_save_dir
        set_global_writer(self.writer)

    @classmethod
    def set_defaults(cls, config):
        def maybe_set(name, value):
            if name not in config:
                config[name] = value

        if 'n_global_iterations' in config:
            maybe_set('n_epochs', 1)
        maybe_set('device', 'cpu')
        maybe_set('log_iteration_scalars', 1)
        maybe_set('log_iteration_all', 'epoch')
        maybe_set('save_iterations', 'epoch')
        maybe_set('prefix', cls.__class__.__name__)
        return config

    @classmethod
    def from_run(cls, _run: sacred.run.Run) -> 'MLProject':
        """ Create a MLProject from a scared ``Run``.  """
        sacred.commands.print_config(_run)
        cfg = _run.config
        return cls(cfg, _id=_run._id, _run=_run)

    @staticmethod
    def get_model(config) -> Model:
        """
        Build a `mlproject.Model` for the given config.
        """
        raise NotImplementedError()

    @staticmethod
    def get_dataset_factory(config) -> DatasetFactory:
        """
        Returns a DatasetFactory from the given config.
        """
        raise NotImplementedError()

    def _is_better(self, score):
        if self.best_score is None:
            return True
        if self.model.minimize_benchmark_metric():
            return self.best_score < score
        else:
            return self.best_score > score

    def test(self):
        # TODO: seperate validation and test
        self.model.eval()
        test_losses = OrderedDict()
        for batch in self.dataset_factory.test_loader():
            losses = self.model.test_batch(batch)
            for name, value in losses.items():
                if name not in test_losses:
                    test_losses[name] = 0
                test_losses[name] += float(to_numpy(value))

        # average over batches
        n_test_batches = len(self.dataset_factory.test_loader())
        for name, loss in test_losses.items():
            test_losses[name] = loss/n_test_batches
        # write and print
        loss_info = ", ".join(["{}: {:.4f}".format(name, float(loss))
                               for name, loss in sorted(test_losses.items())])
        self.writer.add_scalars(self.model.name() + '/test', test_losses, self.global_step)
        print("[TEST] " + loss_info)
        return test_losses[self.model.benchmark_metric()]

    def _should_save(self):
        save_iterations = self.config['save_iterations']
        if self.config['save_iterations'] == 'epoch':
            save_iterations = len(self.dataset_factory.train_loader())
        return (self.epoch_step + 1) % save_iterations == 0

    def _should_stop_training(self):
        if 'n_global_iterations' in self.config:
            return self.global_step >= self.config['n_global_iterations']
        else:
            return self.epoch >= self.config['n_epochs']

    def _iterations_left(self, data_loader_length):
        print('data length', data_loader_length)
        if 'n_global_iterations' in self.config:
            global_iterations = self.config['n_global_iterations']
            iterations_after_epoch = self.global_step + data_loader_length
            if iterations_after_epoch > global_iterations:
                return global_iterations - self.global_step
        return data_loader_length

    def _set_log_level(self):
        log_iteration_scalars = self.config['log_iteration_scalars']
        log_iteration_all = self.config['log_iteration_all']
        if log_iteration_all == 'epoch':
            log_iteration_all = len(self.dataset_factory.train_loader())
        if (self.epoch_step + 1) % log_iteration_scalars == 0:
            if (self.epoch_step + 1) % log_iteration_all == 0:
                self.model.log = LogLevel.ALL
            else:
                self.model.log = LogLevel.SCALARS
        elif (self.epoch_step + 1) % log_iteration_all == 0:
            self.model.log = LogLevel.ALL
        else:
            self.model.log = LogLevel.NONE

    def train(self):
        # TODO: Crtl-C should save the model
        self.model.on_train_begin()
        self.model.train()
        best_model_fname = None

        while True:
            try:
                self.train_epoch()
            except TrainingStop:
                break
            if self.dataset_factory.has_test_set():
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
        if self._should_stop_training():
            raise TrainingStop()
        train_loader = self.dataset_factory.train_loader()
        progbar = tqdm(train_loader, ascii=True,
                       total=self._iterations_left(len(train_loader)))
        self.model.on_epoch_begin(self.epoch)
        self.epoch_step = 0
        for batch in progbar:
            self._set_log_level()
            outs = self.model.train_batch(batch)
            assert not torch.isnan(outs[self.model.benchmark_metric()])
            metrics = {m: outs[m] for m in self.model.metrics() if m in outs}
            if metrics:
                self.writer.add_scalars(self.prefix + '/train', metrics, self.global_step)
            if self._should_save():
                print('model saved:', self.save())
            if self._should_stop_training():
                raise TrainingStop()
            self.global_step += 1
            self.epoch_step += 1

        self.model.on_epoch_end(self.epoch)

    def state_dict(self):
        return {
            '_id': self._id,
            'config': self.config,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'epoch_step': self.epoch_step,
            'best_score': self.best_score,
            'model_save_dir': self.model_save_dir,
            'tensorboard_run_dir': self.tensorboard_run_dir,
            'model_state': self.model.state_dict(),
        }

    def save_filename(self):
        return os.path.abspath(os.path.join(
            self.model_save_dir,
            "{}_e{:05}_b{:05}.torch".format(self.model.name(), self.epoch, self.epoch_step)))

    def save(self):
        torch.save(self.state_dict(), self.save_filename())
        return self.save_filename()

    @classmethod
    def load(cls, filename):
        # TODO: check if training can be continued
        return cls(**torch.load(filename))
