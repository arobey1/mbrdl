import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import torchvision as tv
import torchvision.datasets as datasets

import numpy as np
import math


class BatchTransformDataLoader:
    # Mean normalization on batch level instead of individual
    # https://github.com/NVIDIA/apex/blob/59bf7d139e20fb4fa54b09c6592a2ff862f3ac7f/examples/imagenet/main.py#L222
    def __init__(self, loader, half_prec=True):
        self.loader = loader
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.half_prec = half_prec
        if self.half_prec: self.mean, self.std = self.mean.float(), self.std.float()

    def __len__(self): 
        return len(self.loader)

    def process_tensors(self, input, target, non_blocking=True):
        input = input.cuda(non_blocking=non_blocking).float()

        if len(input.shape) < 3: 
            return input, target.cuda(non_blocking=non_blocking)

        return input.sub_(self.mean).div_(self.std), target.cuda(non_blocking=non_blocking)

    def update_batch_size(self, bs):
        self.loader.batch_sampler.batch_size = bs
            
    def __iter__(self):
        return (self.process_tensors(input, target, non_blocking=True) for input,target in self.loader)

def fast_collate(batch):
    if not batch: return torch.tensor([]), torch.tensor([])
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.tensor(nump_array.copy())
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.tensor(nump_array.copy())
    return tensor, targets

class DistValSampler(Sampler):
    # DistValSampler distrbutes batches equally (based on batch size) to every gpu (even if there aren't enough images)
    # WARNING: Some baches will contain an empty array to signify there aren't enough images
    # Distributed=False - same validation happens on every single gpu
    def __init__(self, indices, args):
        self.indices = indices
        self.batch_size = args.batch_size

        if args.distributed:
            self.world_size = args.world_size
            self.global_rank = args.rank
        else: 
            self.global_rank = 0
            self.world_size = 1
            
        # expected number of batches per sample. Need this so each distributed gpu validates on same number of batches.
        # even if there isn't enough data to go around
        self.expected_num_batches = math.ceil(len(self.indices) / self.world_size / self.batch_size)
        
        # num_samples = total images / world_size. This is what we distribute to each gpu
        self.num_samples = self.expected_num_batches * self.batch_size
        
    def __iter__(self):
        offset = self.num_samples * self.global_rank
        sampled_indices = self.indices[offset:offset+self.num_samples]
        for i in range(self.expected_num_batches):
            offset = i*self.batch_size
            yield sampled_indices[offset:offset+self.batch_size]

    def __len__(self): 
        return self.expected_num_batches

    def set_epoch(self, epoch): 
        return

class BasicImageNetDataset(Dataset):

    def __init__(self, root, args):
        """Dataset for MUNIT ImageNet training."""
        
        xforms = tv.transforms.Compose([
            tv.transforms.Resize((args.data_size, args.data_size)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
        ])
        self._data = datasets.ImageFolder(root, xforms)

    def __getitem__(self, index):
        img, _ = self._data[index]
        return img

    def __len__(self):
        return len(self._data)