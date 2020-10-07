from torch.utils.data import Dataset
import torchvision as tv
import torch
import os
import csv
from PIL import Image


N_GTSRB_CLASSES = 43
GTSRB_CONTRAST_THRESH = [80., 140., 200., 230.]
GTSRB_BRIGHTNESS_THRESH = [40., 85., 125., 170.]            


class GTSRBSubsets(Dataset):
    def __init__(self, mode, challenge, dom, args, return_labels=True):
        """SVHN/GTSRB dataset for training models and classifiers.

        Params:
            mode: Determines kind of data that is returned.
                - choices:  'train' | 'test'.
            challenge: Kind of natural variation.
                - choices: 'brightness' | 'contrast'
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
        self._data = GTSRB(args.train_data_dir, split=split, transform=transform).data

        if self._challenge == 'contrast+brightness':
            self._subsets_dict = self.extract_both()

        else:
            if self._challenge == 'contrast':
                self._thresh = GTSRB_CONTRAST_THRESH
            elif self._challenge == 'brightness':
                self._thresh = GTSRB_BRIGHTNESS_THRESH

            self._subsets_dict, self._values = self.extract_challenge()
        
        if args.local_rank == 0 and args.setup_verbose is True:
            print(f'\tGTSRB {split} set: {[(name, len(ls)) for (name, ls) in self._subsets_dict.items()]}')

    def __getitem__(self, index: int):

        img, label = self._subsets_dict[self._dom][index]
        if self._return_labels is True:
            return img, label
        return img

    def __len__(self) -> int:
        """Number of datapoints in dataset."""

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

            if b <= 50. and c <= 90:
                low.append((img, label))

            elif b >= 170. and c >= 220.:
                high.append((img, label))

        return {'low': low, 'high': high}

class GTSRB:
    def __init__(self, root, split, transform, num_classes=43):
        """GTSRB dataset class to mimic torchvision classes.

        Params:
            root: Root directory for MNIST data.
            split: Which dataset to use.  Choices: 'train' | 'test'.
            transform: Torchvision transforms to apply to data.
            num_classes: Number of classes to keep from GTSRB.
        """

        self._root = root
        self._split = split
        self._transform = transform
        self._num_classes = num_classes

        self._data = self.read_gtsrb_signs()
        # self._data = self.extract_top_kextract_top_k()

    @property
    def data(self):
        return self._data

    def read_gtsrb_signs(self):
        """Reads train sign data for German Traffic Sign Recognition Benchmark.

        Returns:
            List containing images and labels for GTSRB training set.
        """

        def load_from_csv(annotation_fname: str, img_path: str) -> list:
            """Load GTSRB data from annotation_fname.

            Params:
                annotation_fname: Name of CSV file holding names of image files.
                img_path: Base path to images.

            Returns:
                List of torch images and labels.
            """

            curr_data = []
            with open(annotation_fname, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=';')
                next(reader)    # skip header

                for row in reader:
                    img = self._transform(Image.open(os.path.join(img_path, row[0])))
                    curr_data.append((img, int(row[7])))

            return curr_data

        # loop through classes if we want the training split.
        if self._split == 'train':
            data_root = os.path.join(self._root, 'Final_Training', 'Images')

            data = []
            for c in range(0, N_GTSRB_CLASSES):
                class_path = os.path.join(data_root, format(c, '05d'))
                annotation_fname = os.path.join(class_path, 'GT-' + format(c, '05d') + '.csv')
                class_data = load_from_csv(annotation_fname, class_path)
                data.extend(class_data)

        else:
            test_root = os.path.join(self._root, 'Final_Test', 'Images')
            annotation_fname = os.path.join(test_root, 'GT-final_test.csv')
            data = load_from_csv(annotation_fname, test_root)

        return data

    def extract_top_k(self):
        """Extract top k classes from GTSRB.

        Returns:
            List of image label pairs for top k classes.
        """

        if self._num_classes <= 0 or self._num_classes > N_GTSRB_CLASSES:
            raise ValueError(f'k must be an integer between 0 and {N_GTSRB_CLASSES}')

        classes = {c: [] for c in range(N_GTSRB_CLASSES)}
        for (img, label) in self._data:
            classes[label].append(img)

        class_num_imgs = [(c, len(classes[c])) for c in range(N_GTSRB_CLASSES)]
        class_num_imgs.sort(key=lambda x: x[1])
        class_num_imgs.reverse()

        top_k = class_num_imgs[:self._num_classes]
        top_k_classes = sorted([c[0] for c in top_k])
        smallest_num_samples = top_k[-1][1]

        print(f'Top {self._num_classes} classes: {top_k_classes}')
        print(f'Number of images per class: {smallest_num_samples}')

        new_data = []
        for idx, c in enumerate(top_k_classes):
            class_imgs = classes[c][:smallest_num_samples]
            labels_ls = [idx for _ in range(smallest_num_samples)]

            new_data.extend(zip(class_imgs, labels_ls))

        return new_data