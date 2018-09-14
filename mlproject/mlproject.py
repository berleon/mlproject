from collections import OrderedDict
import sacred
from tqdm import tqdm
from mlproject.loader import DatasetLoader
from mlproject.model import Model
from mlproject.log import get_tensorboard_writer, DevNullSummaryWriter, set_global_writer
from mlproject.utils import to_numpy, print_environment_vars


class MLProject:
    def __init__(self, _id, config, dataset_loader: DatasetLoader, model: Model,
                 global_step=0, epoch=0):
        self._id = _id
        self.dataset_loader = dataset_loader
        self.model = model
        self.config = config
        self.global_step = global_step
        self.epoch = epoch
        self.best_score = None
        if self.config.get('tensorboard', False):
            self.writer = get_tensorboard_writer(str(_id) + '_' + self.model.name())
        else:
            self.writer = DevNullSummaryWriter()
        set_global_writer(self.writer)

    @staticmethod
    def from_run(_run: sacred.run.Run, dataset_loader, model_loader):
        sacred.commands.print_config(_run)
        print_environment_vars()
        cfg = _run.config
        return MLProject(_run._id, cfg, dataset_loader(**cfg), model_loader(**cfg))

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
        for epoch_idx in range(self.config['n_train_epochs']):
            self.train_epoch()
            score = self.test()
            if self._is_better(score):
                best_model_fname = self.save()
                self.best_score = score

        if best_model_fname:
            if self.experiment is not None:
                self.experiment.add_artifact(best_model_fname)
        self.model.on_train_end()

    def train_epoch(self):
        progbar = tqdm(self.dataset_loader.train_generator())
        self.model.on_epoch_begin(self.epoch)
        for batch in progbar:
            losses = self.model.train_batch(batch)
            self.writer.add_scalars('training', losses, self.global_step)
            self.global_step += 1
        self.model.on_epoch_end(self.epoch)
        self.epoch += 1

    def save(self):
        # TODO: save project state
        pass
