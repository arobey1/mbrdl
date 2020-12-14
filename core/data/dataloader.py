import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.svhn import SVHNSubsets
from data.cure_tsr import CUREDataset
from data.gtsrb import GTSRBSubsets
from data.mnist import MNISTDataset
from data.mnist_c import MNISTC_Dataset
from data.imagenet import DistValSampler, BatchTransformDataLoader, fast_collate

def get_loaders(args):
    """Load datasets for training/validating classifiers."""

    if args.dataset == 'imagenet':
        return get_imagenet_loaders(args)
    elif args.dataset == 'svhn':
        return get_svhn_loaders(args)
    elif args.dataset == 'cure-tsr' or args.dataset == 'cure_tsr':
        return get_cure_tsr_loaders(args)
    elif args.dataset == 'gtsrb':
        return get_gtsrb_loaders(args)
    elif args.dataset == 'mnist':
        return get_mnist_loaders(args)
    elif args.dataset == 'mnist-c' or args.dataset == 'mnist_c':
        return get_mnist_c_loaders(args)
    else: 
        raise NotImplementedError(f'Dataset {args.dataset} not implemented.')


def get_imagenet_loaders(args):
    """Get dataloaders for ImageNet dataset."""

    train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(args.data_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip()
    ])
    train_dataset = datasets.ImageFolder(args.train_data_dir, train_tfms)
    train_sampler = DistributedSampler(train_dataset) if args.distributed is True else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, collate_fn=fast_collate, 
        sampler=train_sampler)

    val_tfms = transforms.Compose([
        transforms.Resize(int(args.data_size*1.14)), 
        transforms.CenterCrop(args.data_size)
    ])
    val_dataset = datasets.ImageFolder(args.val_data_dir, val_tfms)
    val_sampler = DistValSampler(list(range(len(val_dataset))), args)
    val_loader = DataLoader(val_dataset, num_workers=args.workers, pin_memory=True, 
        collate_fn=fast_collate, batch_sampler=val_sampler)

    train_loader = BatchTransformDataLoader(train_loader, half_prec=args.half_prec)
    val_loader = BatchTransformDataLoader(val_loader, half_prec=args.half_prec)

    return train_loader, val_loader, train_sampler, val_sampler

def get_svhn_loaders(args):
    """Get dataloaders for SVHN dataset."""

    if args.local_rank == 0:
        print(f'Loading SVHN {args.source_of_nat_var} dataset...')

    train_dataset = SVHNSubsets('train', args.source_of_nat_var, dom='low', args=args)
    train_sampler = DistributedSampler(train_dataset) if args.distributed is True else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = SVHNSubsets('test', args.source_of_nat_var, dom='high', args=args)
    val_sampler = DistributedSampler(val_dataset) if args.distributed is True else None
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader, train_sampler, val_sampler

def get_mnist_loaders(args):
    """Get dataloaders for MNIST dataset."""

    if args.local_rank == 0:
        print(f'Loading MNIST {args.source_of_nat_var} dataset...')

    train_dataset = MNISTDataset('train', args.source_of_nat_var, dom='black', args=args)
    train_sampler = DistributedSampler(train_dataset) if args.distributed is True else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = MNISTDataset('test', args.source_of_nat_var, dom='rand', args=args)
    val_sampler = DistributedSampler(val_dataset) if args.distributed is True else None
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader, train_sampler, val_sampler

def get_mnist_c_loaders(args):
    """Get dataloaders for MNIST dataset."""

    if args.local_rank == 0:
        print(f'Loading MNIST-C dataset...')

    train_dataset = MNISTC_Dataset('train', dom='clean', args=args)
    train_sampler = DistributedSampler(train_dataset) if args.distributed is True else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = MNISTC_Dataset('test', dom='corrupted', args=args)
    val_sampler = DistributedSampler(val_dataset) if args.distributed is True else None
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader, train_sampler, val_sampler


def get_cure_tsr_loaders(args):
    """Get dataloaders for CURE-TSR dataset."""

    if args.local_rank == 0:
        print(f'Loading CURE-TSR {args.source_of_nat_var} dataset...')

    train_dataset = CUREDataset('train', args.source_of_nat_var, level=0, args=args)
    train_sampler = DistributedSampler(train_dataset) if args.distributed is True else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = CUREDataset('test', args.source_of_nat_var, level=5, args=args)
    val_sampler = DistributedSampler(val_dataset) if args.distributed is True else None
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader, train_sampler, val_sampler

def get_gtsrb_loaders(args):
    """Get dataloaders for GTSRB dataset."""

    if args.local_rank == 0:
        print(f'Loading GTSRB {args.source_of_nat_var} dataset...')

    train_dataset = GTSRBSubsets('train', args.source_of_nat_var, dom='low', args=args)
    train_sampler = DistributedSampler(train_dataset) if args.distributed is True else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
        shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = GTSRBSubsets('test', args.source_of_nat_var, dom='high', args=args)
    val_sampler = DistributedSampler(val_dataset) if args.distributed is True else None
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader, train_sampler, val_sampler


    



