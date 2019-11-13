import json
import numpy as np
import os
import pickle
import random

NUM_TASKS = 100


class Wiki3029DatasetsBuilder:

    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def get_datasets(self):

        if (self.train_dataset is None
                or self.test_dataset is None
                or self.test_dataset is None):

            train_set, val_set, test_set = self._read_dataset()

            self.train_dataset = Wiki3029Dataset(train_set)
            self.val_dataset = Wiki3029Dataset(val_set)
            self.test_dataset = Wiki3029Dataset(test_set)

        return self.train_dataset, self.val_dataset, self.test_dataset

    def _read_dataset(self):
        """
        Read the Wiki3029 dataset.

        Returns:
          A tuple (train, val, test) of sequences of
            Wiki3029Class instances.
        """
        return tuple(
            self._read_classes(os.path.join(self.data_dir, x))
            for x in ['train', 'val', 'test'])

    def _read_classes(self, dir_path):
        """
        Read the pck files in a directory.
        """
        return [Wiki3029Class(os.path.join(dir_path, f)) for f in os.listdir(dir_path) if f.endswith('.pck')]


class Wiki3029Dataset:

    def __init__(self, dataset, seed=931231):
        np.random.seed(seed)
        self._dataset = dataset

    def get_task(self, num_classes, num_shots):
        x, y = self._generate_task(num_classes, num_shots)
        return {'x': x, 'y': y}

    def _generate_task(self, num_classes, num_shots):
        shuffled = list(self._dataset)
        random.shuffle(shuffled)

        samples = []
        classes = []
        for class_idx, class_obj in enumerate(shuffled[:num_classes]):
            for sample in class_obj.sample(num_shots):
                samples.append(sample)
                classes.append(class_idx)

        return np.array(samples), np.array(classes)


# pylint: disable=R0903
class Wiki3029Class:
    """
    A single article class.
    """
    def __init__(self, dir_path):
        self.file_path = dir_path
        self._cache = None

    def sample(self, num_shots, add=False, normalize=True):
        """
        Sample sentences from the class.

        Returns:
          A sequence of (50,) numpy arrays.
        """
        if self._cache is None:
            self._cache = pickle.load(open(self.file_path, 'rb'))

        tokens_key = 'tokens_norm' if normalize else 'tokens'
        tokens_key = '{}_{}'.format(tokens_key, 'sum' if add else 'mean')

        tokens = self._cache[tokens_key]
        shots = np.random.choice(len(tokens), num_shots, replace=False)
        return tokens[shots]


def generate_tasks_for_set(dataset, set_dict, task_prefix, num_classes, num_shots):
    if not os.path.exists(set_dict):
        os.makedirs(set_dict)

    tasks = []
    for i in range(NUM_TASKS):
        task = dataset.get_task(num_classes, num_shots)
        tasks.append(task)

    users, num_samples, user_data = to_leaf_format(tasks)
    save_json(set_dict, '{}.json'.format(task_prefix), users, num_samples, user_data)


def to_leaf_format(tasks):
    users, num_samples, user_data = [], [], {}

    for i, t in enumerate(tasks):
        x, y = t['x'].tolist(), t['y'].tolist()
        u_id = str(i)

        users.append(u_id)
        num_samples.append(len(y))
        user_data[u_id] = {'x': x, 'y': y}

    return users, num_samples, user_data


def save_json(json_dir, json_name, users, num_samples, user_data):
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    json_file = {
        'users': users,
        'num_samples': num_samples,
        'user_data': user_data,
    }

    with open(os.path.join(json_dir, json_name), 'w') as outfile:
        json.dump(json_file, outfile)


def main():
    random.seed(931231)
    wiki3029_set_builder = Wiki3029DatasetsBuilder('./wiki3029_pck')
    train_set, val_set, test_set = wiki3029_set_builder.get_datasets()

    generate_tasks_for_set(train_set, 'wiki3029_tasks/5way_10shot/train', 'train', num_classes=5, num_shots=10)
    generate_tasks_for_set(val_set, 'wiki3029_tasks/5way_10shot/val', 'val', num_classes=5, num_shots=10)
    generate_tasks_for_set(test_set, 'wiki3029_tasks/5way_10shot/test', 'test', num_classes=5, num_shots=10)


if __name__ == '__main__':
    main()
