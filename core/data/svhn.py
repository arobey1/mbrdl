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
                - choices: 'brightness' | 'contrast' | 'contrast+brightness'
            dom: Domain to be used.
                - choices: 'low' | 'medium' | 'high'
            args: Command line arguments.
            return_labels: If True, returns image and label.  Otherwise
                only returns image (without label).
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
        self._data = tv.datasets.SVHN(args.train_data_dir, split=split, transform=transform, download=False)

        if self._challenge == 'contrast+brightness':
            self._subsets_dict = self.extract_both()

        else:
            if self._challenge == 'contrast':
                self._thresh = SVHN_CONTRAST_THRESH
            elif self._challenge == 'brightness':
                self._thresh = SVHN_BRIGHTNESS_THRESH

            self._subsets_dict, self._values = self.extract_challenge()

        if args.local_rank == 0 and args.setup_verbose is True:
            print(f'\tSVHN {split} set: {[(name, len(ls)) for (name, ls) in self._subsets_dict.items()]}')

    def __getitem__(self, index: int):

        img, label = self._subsets_dict[self._dom][index]
        if self._return_labels is True:
            return img, label
        return img

    def __len__(self) -> int:
        """Returns number of datapoints in dataset."""

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
        """Extract both subsets based on contrast and brightness."""

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
