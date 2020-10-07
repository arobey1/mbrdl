from torch.utils.data import Dataset
import torchvision as tv
import torchvision.datasets as datasets
import random
from PIL import Image
import re
import torch
import os
import glob

class CUREDataset(Dataset):
    def __init__(self, mode, challenge, level, args, return_labels=True):

        self._mode = mode
        self._challenge = challenge.capitalize()
        self._level = level
        self._return_labels = return_labels
        self._dom = f'{self._challenge}-{level}'

        transform = tv.transforms.Compose([
            tv.transforms.Resize((args.data_size, args.data_size)),
            tv.transforms.ToTensor()
        ])

        split = 'Real_Test' if 'test' in self._mode else 'Real_Train'
        self._data = self.__load_all_data(args.train_data_dir, split, transform)

    def __getitem__(self, index):

        img, label = self._data[self._dom][index]

        if self._return_labels is True:
            return img, label
        return img

    def __len__(self):

        return len(self._data[f'{self._challenge}-{self._level}'])

    @staticmethod
    def __load_dataset(root_dir, transform, cf=False):
        """Load individual CURE dataset from files.

        Params:
            root_dir: Root directory for CURE challenge data subset.
            transform: Torch transforms to apply to data.
            cf: If cf (challenge free) is True, loads challenge free labels.

        Returns:
            List with (img, label) data.
        """

        files = sorted(glob.glob(root_dir + '/*.*'))
        imgs = [transform(Image.open(fname)) for fname in files]

        if cf is True:
            labels = [int(re.findall(r'\d+', fname)[1]) - 1 for fname in files]
        else:
            labels = [int(re.findall(r'\d+', fname)[2]) - 1 for fname in files]

        data = list(zip(imgs, labels))
        random.shuffle(data)

        return data

    def __load_all_data(self, root: str, split: str, transform) -> dict:
        """Load all CURE datasets from file.

        Params:
            root: Root directory of CURE data.
            split: Directory name for training/test split.
            transform: Torch transforms to apply to data.

        Returns:
            Dictionary of datasets corresponding to different challenge levels.
        """

        datasets = {}

        chall_free_path = os.path.join(root, split, 'ChallengeFree')
        datasets[f'{self._challenge}-0'] = self.__load_dataset(chall_free_path, transform, cf=True)

        if self._level != 0:
            chall_path = os.path.join(root, split, f'{self._challenge}-{self._level}')
            datasets[f'{self._challenge}-{self._level}'] = self.__load_dataset(chall_path, transform, cf=False)

        return datasets


class CURE:
    def __init__(self, mode, challenge, level, args):
        self._mode = mode
        self._dom = f'{self._challenge.capitalize()}-{level}'
        self._level = level

        root = './datasets/cure_tsr/raw_data'
        transform = tv.transforms.Compose([
            tv.transforms.Resize((args.data_size, args.data_size)),
            tv.transforms.ToTensor()
        ])

        split = 'Real_Test' if 'test' in self._mode else 'Real_Train'
        self._data = self.__load_all_data(root, split, transform)

    @property
    def data(self):
        return self._data

    @staticmethod
    def __load_dataset(root_dir, transform, cf=False):
        """Load individual CURE dataset from files.

        Params:
            root_dir: Root directory for CURE challenge data subset.
            transform: Torch transforms to apply to data.
            cf: If cf (challenge free) is True, loads challenge free labels.

        Returns:
            List with (img, label) data.
        """

        files = sorted(glob.glob(root_dir + '/*.*'))
        imgs = [transform(Image.open(fname)) for fname in files]

        if cf is True:
            labels = [int(re.findall(r'\d+', fname)[1]) - 1 for fname in files]
        else:
            labels = [int(re.findall(r'\d+', fname)[2]) - 1 for fname in files]

        data = list(zip(imgs, labels))
        random.shuffle(data)

        return data

    def __load_all_data(self, root: str, split: str, transform) -> dict:
        """Load all CURE datasets from file.

        Params:
            root: Root directory of CURE data.
            split: Directory name for training/test split.
            transform: Torch transforms to apply to data.

        Returns:
            Dictionary of datasets corresponding to different challenge levels.
        """

        if self._level == 0:
            path = os.path.join(root, split, 'ChallengeFree')
            return self.__load_dataset(path, transform, cf=True)
        else:
            path = os.path.join(root, split, self._dom)
            return self.__load_dataset(path, transform, cf=False)
