from torch.utils.data import Dataset
import torchvision as tv
import torch

SVHN_CONTRAST_THRESH = [80, 90, 100, 190]
SVHN_BRIGHTNESS_THRESH = [60., 160., 170., 180.]

class SVHNSubsets(Dataset):
    def __init__(self, mode, challenge, dom, args, return_labels=True):
        """SVHN/GTSRB dataset for training models and classifiers.

        Params:
            mode: Determines kind of data that is returned.
                - choices:  'train' | 'test'.
            challenge: Kind of natural variation.
                - choices: 'brightness' | 'contrast'
        """

        self._mode = mode
        self._challenge = challenge
        self._dom = dom            
        self._return_labels = return_labels

        transform = tv.transforms.Compose([
            tv.transforms.Resize((args.data_size, args.data_size)),
            tv.transforms.ToTensor()
        ])

        split = 'test' if 'test' in self._mode else 'train'
        self._data = tv.datasets.SVHN('./datasets/svhn', split=split, transform=transform, download=False)

        if self._challenge == 'contrast+brightness':
            self._subsets_dict = self.extract_both()

        else:
            if self._challenge == 'contrast':
                self._thresh = SVHN_CONTRAST_THRESH
            elif self._challenge == 'brightness':
                self._thresh = SVHN_BRIGHTNESS_THRESH

            self._subsets_dict, self._values = self.extract_challenge()

        # print([(name, len(ls)) for (name, ls) in self._subsets_dict.items()])

    def __getitem__(self, index: int):

        img, label = self.mod_index(index, self._dom)

        if self._return_labels is True:
            return img, label
        return img

    def __len__(self) -> int:
        """Returns length of dataset.  If self._num_items is set, this number is
        returned.  Otherwise, we return the smallest length of the low, medium,
        and high subsets."""

        return len(self._subsets_dict[self._dom])

    def extract_challenge(self) -> dict:
        """Extract data subsets from dataset.

        Returns:
            Dictionary containing data subsets for 'high', 'medium', 'low', 'all'.
        """

        if self._challenge == 'contrast':
            chall_fn = lambda x: 255 * (torch.max(x) - torch.min(x))

        elif self._challenge == 'brightness':
            chall_fn = lambda x: torch.mean(x) * 255.

        low, medium, high, values = ([] for _ in range(4))

        # TODO: parfor
        for (img, label) in self._data:

            val = chall_fn(img)
            values.append(val)

            if val <= self._thresh[0]:
                low.append((img, label))
            elif self._thresh[1] <= val <= self._thresh[2]:
                medium.append((img, label))
            elif val >= self._thresh[3]:
                high.append((img, label))

        return {'low': low, 'medium': medium, 'high': high, 'all': self._data}, values

    def extract_both(self):

        contrast_fn = lambda x: 255 * (torch.max(x) - torch.min(x))
        brightness_fn = lambda x: torch.mean(x) * 255.

        low, high = ([] for _ in range(2))
        for (img, label) in self._data:

            b, c = brightness_fn(img), contrast_fn(img)

            if b <= 70. and c <= 90:
                low.append((img, label))

            elif b >= 170. and c >= 180.:
                high.append((img, label))

        return {'low': low, 'high': high}

    def mod_index(self, index: int, name: str):
        """Ensures that we never get an IndexError when indexing into a data subset.

        Params:
            index: Index of datum requrested by __getitem__ method.
            name: Name of data subset.  Choices: 'low', 'medium', 'high'.

        Returns:
            (img, label) pair at index that will not cause error.
        """

        data_list = self._subsets_dict[name]
        return data_list[index % len(data_list)]
