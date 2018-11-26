import numpy as np

import os
import copy
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os.path as path
import os
import errno
import pickle


class ClutteredMNIST:
    # TODO: Export dataset
    def __init__(self, dataset, shape=(100, 100), n_clutters=6, clutter_size=8,
                 n_samples=60000, transform=None):
        self.dataset = dataset
        self.shape = shape
        self.n_clutters = n_clutters
        self.clutter_size = clutter_size
        self.n_samples = n_samples
        self.transform = transform
        self._parameters = None  # are set on self._init() to save time when they are not needed

    def get_parameters(self):
        if self._parameters is None:
            self._init_parameters()
        return self._parameters

    def _init_parameters(self):
        self._parameters = self.generate_parameters()

    def export(self, datadir, force_write=False) -> bool:
        """
        write image files in the file system, which can be loaded with:
        https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder
        or do nothing if they already exist
        structure: [datadir]/[class idx]/[sample idx].png
        :param force_write: if True, the files are written, even if they already exist
        :param datadir: the location for the data
        :return: boolean, True if the data was written, False if the data was already there
        """
        # TODO check if files are complete
        # TODO remove MNIST data after export completion
        # TODO on overwriting, remove all existing files first (prevent leftovers from old dataset)
        abspath = path.abspath(datadir)
        meta = {
            "n_samples": self.n_samples,
            "clutter_size": self.clutter_size,
            "n_clutters": self.n_clutters,
        }
        metapath = path.join(abspath, 'meta.p')
        if path.exists(metapath) and not force_write:
            # this seems to be a existing dataset folder. if it is identical, we can reuse it
            with open(metapath, mode='rb') as fp:
                existing_meta = pickle.load(fp)
                # check if meta is identical
                for name, item in meta.items():
                    if name not in existing_meta:
                        raise RuntimeError("There is already a dataset stored in this location "
                                           "and it does not specify the config value {}".format(name))
                    if existing_meta[name] != meta[name]:
                        raise RuntimeError("There is already a dataset stored in this location "
                                           "and the config value {} differs: {} != {}. Please "
                                           "remove the files or choose different location."
                                           "".format(name, existing_meta[name], meta[name]))
            # the files exist and are identical. we can quit.
            return False

        # the files do not yet exist. we will write them
        print("ClutteredMNIST: Exporting {} files to {}".format(self.n_samples, abspath))
        # write image files
        tmp_transform = self.transform  # store transform
        self.transform = None
        counters = list()
        for i, (pil_img, label) in enumerate(self):
            if isinstance(label, torch.Tensor):
                label = label.detach().cpu().numpy()
            while len(counters) <= label:
                counters.append(0)
            counters[label] += 1
            labelpath = self.ensure_dir(path.join(abspath, str(label)))
            pil_img.save(path.join(labelpath, str(counters[label])+".png"), format="png")
        self.transform = tmp_transform  # restore transform
        # write meta
        # add params to meta (we did not want to generate them earlier, in case we didn't need them)
        meta["params"] = self.get_parameters()
        with open(metapath, 'wb') as fp:
            pickle.dump(meta, fp, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def ensure_dir(dirname) -> str:
        """ check if dir exists, otherwise create it safely or raise error"""
        try:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        return dirname

    def generate_parameters(self):
        all_params = []
        h, w = self.dataset[0][0].size
        for i in range(self.n_samples):
            params = {
                'idx': i % len(self.dataset),
                'digit_h': np.random.randint(0, self.shape[0] - h),
                'digit_w': np.random.randint(0, self.shape[1] - w),
            }
            clutter = []
            for _ in range(self.n_clutters):
                clutter_idx = np.random.randint(0, len(self.dataset))
                cs = self.clutter_size
                ph = np.random.randint(0, h - cs)
                pw = np.random.randint(0, w - cs)
                ch = np.random.randint(0, self.shape[0] - cs)
                cw = np.random.randint(0, self.shape[1] - cs)
                clutter.append({
                    'clutter_idx': clutter_idx,
                    'patch_h': ph,
                    'patch_w': pw,
                    'clutter_h': ch,
                    'clutter_w': cw,
                })
            params['clutter'] = clutter
            all_params.append(params)
        return all_params

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self._parameters is None:
            self._init_parameters()
        canvas = np.zeros(self.shape, dtype=np.uint8)
        params = self._parameters[idx]
        for clutter in params['clutter']:
            clutter_img = np.array(self.dataset[clutter['clutter_idx']][0])
            h, w = clutter_img.shape
            # select patch
            cs = self.clutter_size
            ph = clutter['patch_h']
            pw = clutter['patch_w']
            patch = clutter_img[ph:ph+cs, pw:pw+cs]
            # place patch
            ch = clutter['clutter_h']
            cw = clutter['clutter_w']
            canvas[ch:ch+cs, cw:cw+cs] = patch

        img, label = self.dataset[params['idx']]
        img = np.array(img)
        h, w = img.shape
        dh = params['digit_h']
        dw = params['digit_w']
        canvas[dh:dh+h, dw:dw+w] = img
        pil_img = Image.fromarray(canvas, mode='L')
        if self.transform is not None:
            return self.transform(pil_img), label
        else:
            return pil_img, label


class DatasetFactory:
    """
    The DatasetFactory provides access to the train/test/val datasets and all the dataset
    iterators.

    The `Dataset`'s returned by `train_set`, `test_set`, and `validation_set`
    should be persistent, e.g.  an index should always return the same data and
    label.

    The `DataLoaders` returned by *_loader can of course return the data
    augmented and in any order.
    """
    def __init__(self,
                 train_set=None, train_loader=None,
                 test_set=None, test_loader=None,
                 validation_set=None, validation_loader=None):
        self._train_set = train_set
        self._train_loader = train_loader
        self._test_set = test_set
        self._test_loader = test_loader
        self._validation_set = validation_set
        self._validation_loader = validation_loader

    def train_set(self) -> Dataset:
        """Return the train set."""
        return self._train_set

    def train_loader(self) -> DataLoader:
        """Return the DataLoader associated with the train set."""
        return self._train_loader

    def test_set(self) -> Dataset:
        """Return the test set."""
        return self._test_set

    def test_loader(self) -> DataLoader:
        """Return the DataLoader associated with the test set."""
        return self._test_loader

    def validation_set(self) -> Dataset:
        """Return the validation set."""
        return self._validation_set

    def validation_loader(self) -> DataLoader:
        """Return the DataLoader associated with the validation set."""
        return self._validation_loader

    def has_train_set(self) -> bool:
        return self.train_set() is not None

    def has_test_set(self) -> bool:
        return self.test_set() is not None

    def has_validation_set(self) -> bool:
        return self.validation_set() is not None


class CycleDataLoader(DataLoader):
    def __init__(self, dataloader, n_cycles=1000):
        self.dataloader = dataloader
        self.n_cycles = n_cycles

    def __iter__(self):
        for _ in range(self.n_cycles):
            for batch in self.dataloader:
                yield batch

    def __len__(self):
        return len(self.dataloader) * self.n_cycles


class CycleDatasetFactory(DatasetFactory):
    def __init__(self, factory: DatasetFactory, n_cycles=1000):
        def cycle(loader):
            if loader is None:
                return None
            return CycleDataLoader(loader, n_cycles)
        super().__init__(
            factory.train_set(),
            cycle(factory.train_loader()),
            factory.test_set(),
            cycle(factory.test_loader()),
            factory.validation_set(),
            cycle(factory.validation_loader()),
        )


def default_data_dir(maybe_data_dir=None):
    if maybe_data_dir is not None:
        return maybe_data_dir
    elif "DATA_DIR" in os.environ:
        return os.environ['DATA_DIR']
    else:
        raise ValueError("Can not figure out data_dir. "
                         "Please set the DATA_DIR enviroment variable.")


class TorchvisionDatasetFactory(DatasetFactory):
    # TODO: Extract more code from subclass
    def __init__(self, train_set=None, test_set=None, validation_set=None,
                 data_loader_kwargs={},
                 data_loader_train_kwargs={},
                 data_loader_test_kwargs={}):
        train_kwargs = copy.copy(data_loader_kwargs)
        train_kwargs.update(data_loader_train_kwargs)
        test_kwargs = copy.copy(data_loader_kwargs)
        test_kwargs.update(data_loader_test_kwargs)

        if train_set is not None:
            trainloader = torch.utils.data.DataLoader(train_set, **train_kwargs)
        else:
            trainloader = None

        if test_set is not None:
            testloader = torch.utils.data.DataLoader(test_set, **test_kwargs)
        else:
            testloader = None

        if validation_set is not None:
            valloader = torch.utils.data.DataLoader(validation_set, **test_kwargs)
        else:
            valloader = None

        super().__init__(
            train_set, trainloader,
            test_set, testloader,
            validation_set, valloader,
        )


class CIFARDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size=50, train_transform=None, test_transform=None, data_dir=None,
                 num_workers=0):
        self.data_dir = default_data_dir(data_dir)
        trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                               download=True, transform=test_transform)
        super().__init__(
            trainset, testset,
            data_loader_kwargs={
                'num_workers': num_workers,
                'batch_size': batch_size,
            },
            data_loader_train_kwargs={'shuffle': True}
        )


class FashionMNISTDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size=1, train_transform=None, test_transform=None, data_dir=None,
                 num_workers=0, collate_fn=None, pin_memory=None, drop_last=None,
                 data_loader_train_kwargs=None,
                 data_loader_test_kwargs=None,
                 ):
        def update_kwargs(kwargs):
            kwargs = copy.copy(kwargs or {})
            if 'num_workers' not in kwargs:
                kwargs['num_workers'] = num_workers
            if 'collate_fn' not in kwargs and collate_fn is not None:
                kwargs['collate_fn'] = collate_fn
            if 'pin_memory' not in kwargs and pin_memory is not None:
                kwargs['pin_memory'] = pin_memory
            if 'drop_last' not in kwargs and drop_last is not None:
                kwargs['drop_last'] = drop_last
            if 'batch_size' not in kwargs:
                kwargs['batch_size'] = batch_size
            return kwargs

        self.data_dir = default_data_dir(data_dir)
        trainset = torchvision.datasets.FashionMNIST(root=self.data_dir, train=True,
                                                     download=True, transform=train_transform)
        testset = torchvision.datasets.FashionMNIST(root=self.data_dir, train=False,
                                                    download=True, transform=test_transform)
        super().__init__(
            trainset, testset,
            data_loader_train_kwargs=update_kwargs(data_loader_train_kwargs),
            data_loader_test_kwargs=update_kwargs(data_loader_test_kwargs)
        )


class MNISTDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size=50, train_transform=None, test_transform=None, data_dir=None,
                 num_workers=0):
        self.data_dir = default_data_dir(data_dir)
        trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True,
                                              download=True, transform=train_transform)
        testset = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                             download=True, transform=test_transform)
        super().__init__(
            trainset, testset,
            data_loader_kwargs={
                'num_workers': num_workers,
                'batch_size': batch_size,
            },
            data_loader_train_kwargs={'shuffle': True}
        )


class ClutteredMNISTDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size=50, train_transform=None, test_transform=None, data_dir=None,
                 shape=(100, 100), n_clutters=6, clutter_size=8, n_samples_train=60000, n_samples_test=10000, n_samples_val=10000,
                 num_workers=0, use_filesys=False):
        self.data_dir = default_data_dir(data_dir)
        # create dataset generators
        trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True)
        testset = torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True)
        generator_train = ClutteredMNIST(trainset, shape, n_clutters,
                                         clutter_size, n_samples_train,
                                         transform=train_transform)
        generator_test = ClutteredMNIST(testset, shape, n_clutters,
                                        clutter_size, n_samples_test,
                                        transform=test_transform)
        if use_filesys:
            # export dataset files to data dir
            train_dir = path.join(data_dir, "train")
            test_dir = path.join(data_dir, "test")
            generator_train.export(train_dir)
            generator_test.export(test_dir)

            # create Torchvision ImageFolder Dataset of these dirs
            def loader(path) -> Image:
                return Image.open(path).convert("L")  # force one channel, just like original MNIST
            resource_train = torchvision.datasets.ImageFolder(train_dir, transform=train_transform, loader=loader)
            resource_test = torchvision.datasets.ImageFolder(test_dir, transform=test_transform, loader=loader)
        else:
            # use just-in-time generation
            resource_train = generator_train
            resource_test = generator_test

        # construct parent
        super().__init__(
            resource_train, resource_test,
            data_loader_kwargs={
                'num_workers': num_workers,
                'batch_size': batch_size,
            },
            data_loader_train_kwargs={'shuffle': True}
        )
