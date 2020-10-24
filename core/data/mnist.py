from torch.utils.data import Dataset
import torchvision as tv
import torch

COLOR_DICT = {
    'red': [1., 0., 0.],
    'blue': [0., 1., 0.],
    'green': [0., 0., 1.],
}

class MNISTDataset(Dataset):

    def __init__(self, mode, challenge, dom, args, return_labels=True):

        self._mode = mode
        self._challenge = challenge
        self._dom = dom
        self._return_labels = return_labels

        transform = tv.transforms.Compose([
            tv.transforms.Resize((args.data_size, args.data_size)),
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

        train = True if 'train' in self._mode else False
        self._data = tv.datasets.MNIST(args.train_data_dir, train=train, 
                                        download=True, transform=transform)
        self._data = self.colorize_dataset()

    def __getitem__(self, index):
        img, label = self._data[index]
        if self._return_labels is True:
            return img, label
        return img

    def __len__(self):
        return len(self._data)
    
    def colorize_dataset(self):

        def colorize_img(img, color_name):

            if color_name in COLOR_DICT.keys():
                color = COLOR_DICT[color_name]
            elif color_name == 'rand':
                color = torch.rand(3)
            else:
                raise ValueError('Invalid color.')

            zero_tensor = torch.zeros_like(img)
            zero_tensor[0, :, :] = color[0]
            zero_tensor[1, :, :] = color[1]
            zero_tensor[2, :, :] = color[2]

            return torch.where(img < 0.05, zero_tensor, img)

        if self._dom == 'black':
            return self._data

        new_data = []
        for i in range(len(self._data)):
            img, label = self._data[i]
            new_img = colorize_img(img, self._dom)
            new_data.append((new_img, label))

        new_data.extend(self._data)

        return new_data