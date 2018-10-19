import pickle
import os
import sys
import pprint
import tempfile
import warnings

import gridfs
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import torch
from sacred import Experiment
from sacred.observers import MongoObserver


def add_mongodb(ex: Experiment):
    ex.observers.append(MongoObserver.create(url=get_mongodb_ip()))


def add_package_sources(ex: Experiment):
    this_dir = os.path.dirname(__file__)
    package_dirs = [this_dir, os.path.join(this_dir, "..")]
    for package_dir in package_dirs:
        for name in os.listdir(package_dir):
            if name.endswith(".py"):
                ex.add_source_file(os.path.abspath(os.path.join(package_dir, name)))


def load_weights(model, artifact_filename, mode='eval'):
    # pytorch needs a "real" (fileno) file
    with open(artifact_filename, mode='rb') as f:
        model.load_state_dict(torch.load(f))
        if mode == 'eval':
            model.eval()
        else:
            model.train()
        model.cuda()


def get_db(database_name='sacred'):
    warnings.warn("This code needs some cleanup. Tell me before you want to use it.")
    client = MongoClient(host=get_mongodb_ip())
    db = client.get_database(database_name)
    return db, gridfs.GridFS(db)


def get_mongodb_ip(config_dir="~/.config/docker_ports/"):
    # TODO: use env for mongodb ip
    config_dir = os.path.expanduser(config_dir)
    filename = os.path.join(config_dir, "docker_mongodb_ip")
    with open(filename) as f:
        return f.read()


def get_id(find, sort, database_name='sacred'):
    print(find)
    print(sort)
    db, _ = get_db(database_name)
    result = db.runs.find_one(find, projection={'_id': 1}, sort=sort)
    return result['_id']


def load_entry(id, database_name='sacred'):
    db, _ = get_db(database_name)
    run_entry = db.runs.find_one({'_id': id})
    return run_entry


def load_experiment(experiment, id, database_name='sacred'):
    run_entry = load_entry(id)
    return experiment._create_run(config_updates=run_entry['config']), run_entry


def weight_files(db_entry):
    weights = []
    for artifact in db_entry['artifacts']:
        name = artifact['name']
        if name.endswith('weight'):
            try:
                iteration = int(name.split('.')[0].split('_')[-1])
            except ValueError:
                iteration = None
            weights.append((iteration, artifact))
    return sorted(weights, key=lambda w: w[0] or 0)


def load_model(db_entry):
    raise Exception()
    if type(db_entry) == int:
        db_entry = load_entry(db_entry)

    # TODO: fix this
    #
    iteration, latest_weight = weight_files(db_entry)[-1]
    load_weights_from_db(model, latest_weight['file_id'])
    return model


def load_weights_from_db(model, file_id=None, db_entry=None, database_name='sacred'):
    if file_id is None:
        iteration, latest_weight = weight_files(db_entry)[-1]
        file_id = latest_weight['file_id']
    _, fs = get_db(database_name)
    f = fs.get(file_id)
    with tempfile.NamedTemporaryFile() as tmpf:
        tmpf.write(f.read())
        tmpf.flush()
        load_weights(model, tmpf.name)
    f.close()


def print_experiment(ex, stream=None, skip=['captured_out', 'artifacts', 'results']):
    if stream is None:
        stream = sys.stdout
    pp = pprint.PrettyPrinter(indent=2, stream=stream)
    for k, v in sorted(ex.items()):
        if k in skip:
            continue
        stream.write('\n')
        stream.write("-" * 40 + " " + k + " " + "-" * 40 + "\n")
        pp.pprint(v)


def get_metric(ex, name, db=None):
    if db is None:
        db, _ = get_db()

    for metric in ex['info']['metrics']:
        if metric['name'] == name:
            return db.metrics.find_one({'_id': ObjectId(metric['id'])})

    raise KeyError("No metric named {} found".format(name))


def yield_metrics(ex, marker=None, db=None):
    if db is None:
        db, _ = get_db()
    if 'info' not in ex or 'metrics' not in ex['info']:
        return
    for metric in sorted(ex['info']['metrics'], key=lambda m: m['name']):
        print(metric)
        if marker is not None and marker not in metric['name']:
            continue
        yield db.metrics.find_one({'_id': ObjectId(metric['id'])})


class ResultStorage:
    def __init__(self, run_id, iteration):
        self.db, self.gridfs = get_db()
        self.run_id = run_id
        self.iteration = iteration

        # self.db.runs.find_one_and_update(
        #     {'_id': self.run_id},
        #     {'results': {str(self.iteration): []}}
        # )

    def gridfs_filename(self, name):
        return 'results://{}/{}/{}'.format(self.run_id, self.iteration, name)

    @property
    def iteration_key(self):
        return 'results.{}'.format(self.iteration)

    def result_key(self, name):
        return '{}.{}'.format(self.iteration_key, name)

    def get_result(self, name):
        entry = self.db.runs.find_one(
            {'_id': self.run_id},
            {self.result_key(name): 1}
        )
        if ('results' in entry and str(self.iteration) in entry['results']
                and name in entry['results'][str(self.iteration)]):
            return entry['results'][str(self.iteration)][name]

    def store(self, name, value):
        self.invalidate(name)

        f = self.gridfs.new_file(filename=self.gridfs_filename(name))
        f.write(pickle.dumps(value))
        f.close()

        self.db.runs.update_one(
            {'_id': self.run_id},
            {
                '$set': {self.result_key(name): {
                    'file_id': f._id,
                    'create_at': datetime.utcnow(),
                    'name': name
                }},
            }
        )

    def get(self, name):
        result = self.get_result(name)
        if result is None:
            raise KeyError("No result under {}".format(name))
        grid_file = self.gridfs.get(result['file_id'])
        return pickle.loads(grid_file.read())

    def cache(self, name, lazy_value):
        try:
            return self.get(name)
        except KeyError:
            value = lazy_value()
            self.store(name, value)
            return value

    def invalidate(self, name):
        result = self.get_result(name)
        self.db.runs.update_one({'_id': self.run_id},
                                {'$unset': {self.result_key(name): {}}})
        if result is not None:
            self.gridfs.delete(result['file_id'])

    def all_results(self):
        entry = self.db.runs.find_one({'_id': self.run_id},
                                      {'results.{}'.format(self.iteration): 1})
        return entry['results'][str(self.iteration)]

    def invalidate_all(self):
        for name, result in self.all_results().items():
            self.gridfs.delete(result['file_id'])
        self.db.runs.update_one({'_id': self.run_id},
                                {'$unset': {self.iteration_key: {}}})

        prefix_regex = "^{}.*".format(self.gridfs_filename(''))

        for entry in self.db.fs.files.find({'filename': {'$regex': prefix_regex}}):
            warnings.warn("Deleting orphan result: " + str(entry))
            self.gridfs.delete(entry['_id'])
