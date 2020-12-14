import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.svhn import SVHNSubsets
from data.cure_tsr import CUREDataset
from data.gtsrb import GTSRBSubsets
from data.mnist import MNISTDataset
from data.imagenet import BasicImageNetDataset
from data.mnist_c import MNISTC_Dataset

def get_munit_loaders(args):
    """Return dataloaders based on command line argument for dataset."""

    if args.dataset == 'svhn':
        return get_svhn_loaders(args)
    elif args.dataset == 'gtsrb':
        return get_gtsrb_loaders(args)
    elif args.dataset == 'cure-tsr' or args.dataset == 'cure_tsr':
        return get_cure_tsr_loaders(args)
    elif args.dataset == 'imagenet':
        return get_imagenet_loaders(args)
    elif args.dataset == 'mnist':
        return get_mnist_loaders(args)
    elif args.dataset == 'mnist-c' or args.dataset == 'mnist_c':
        return get_mnist_c_loaders(args)
    else:
        raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

def get_imagenet_loaders(args):
    """Get loaders with ImageNet/ImageNet-c data for training MUNIT."""

    if args.local_rank == 0:
        print(f'Loading ImageNet dataset...')

    train_A = test_A = BasicImageNetDataset(args.train_data_dir, args=args)
    train_B = test_B = BasicImageNetDataset(args.val_data_dir, args=args)

    return _to_loader(train_A, train_B, test_A, test_B)

def get_mnist_loaders(args):
    """Get loaders with MNIST data for training MUNIT."""

    if args.local_rank == 0:
        print(f'Loading MNIST {args.source_of_nat_var} dataset...')

    train_A = MNISTDataset('train', args.source_of_nat_var, dom='black',
                            args=args, return_labels=False)
    train_B = MNISTDataset('train', args.source_of_nat_var, dom='rand',
                            args=args, return_labels=False)
    test_A = MNISTDataset('test', args.source_of_nat_var, dom='black',
                            args=args, return_labels=False)
    test_B = MNISTDataset('test', args.source_of_nat_var, dom='rand',
                            args=args, return_labels=False)

    return _to_loader(train_A, train_B, test_A, test_B)

def get_mnist_c_loaders(args):
    
    if args.local_rank == 0:
        print(f'Loading MNIST-C dataset...')

    train_A = MNISTC_Dataset('train', dom='clean', args=args, return_labels=False)
    train_B = MNISTC_Dataset('train', dom='corrupted', args=args, return_labels=False)

    test_A = MNISTC_Dataset('test', dom='clean', args=args, return_labels=False)
    test_B = MNISTC_Dataset('test', dom='corrupted', args=args, return_labels=False)

    return _to_loader(train_A, train_B, test_A, test_B)

def get_svhn_loaders(args):
    """Get loaders with SVHN data for training MUNIT."""

    if args.local_rank == 0:
        print(f'Loading SVHN {args.source_of_nat_var} dataset...')

    train_A = SVHNSubsets('train', args.source_of_nat_var, dom='low', 
                            args=args, return_labels=False)
    train_B = SVHNSubsets('train', args.source_of_nat_var, dom='high',
                            args=args, return_labels=False)
    test_A = SVHNSubsets('test', args.source_of_nat_var, dom='low', 
                            args=args, return_labels=False)
    test_B = SVHNSubsets('test', args.source_of_nat_var, dom='high',
                            args=args, return_labels=False)

    return _to_loader(train_A, train_B, test_A, test_B)

def get_gtsrb_loaders(args):
    """Get loaders with GTSRB data for training MUNIT."""

    if args.local_rank == 0:
        print(f'Loading GTSRB {args.source_of_nat_var} dataset...')

    train_A = GTSRBSubsets('train', args.source_of_nat_var, dom='low', 
                            args=args, return_labels=False)
    train_B = GTSRBSubsets('train', args.source_of_nat_var, dom='high',
                            args=args, return_labels=False)
    test_A = GTSRBSubsets('test', args.source_of_nat_var, dom='low', 
                            args=args, return_labels=False)
    test_B = GTSRBSubsets('test', args.source_of_nat_var, dom='high',
                            args=args, return_labels=False)

    return _to_loader(train_A, train_B, test_A, test_B)

def get_cure_tsr_loaders(args):
    """Get loaders with CURE-TSR data for training MUNIT."""

    if args.local_rank == 0:
        print(f'Loading CURE-TSR {args.source_of_nat_var} dataset...')

    train_A = CUREDataset('train', args.source_of_nat_var, level=0,
                            args=args, return_labels=False)
    train_B = CUREDataset('train', args.source_of_nat_var, level=5,
                            args=args, return_labels=False)
    test_A = CUREDataset('test', args.source_of_nat_var, level=0,
                            args=args, return_labels=False)
    test_B = CUREDataset('test', args.source_of_nat_var, level=5,
                            args=args, return_labels=False)

    return _to_loader(train_A, train_B, test_A, test_B)

def _to_loader(*args):
    """Turn datasets into dataloaders for MUNIT."""

    def make_loader(dataset):
        return DataLoader(dataset, batch_size=1, shuffle=True)

    return [make_loader(d) for d in args]
        