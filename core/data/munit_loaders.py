import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.svhn import SVHNSubsets
from data.cure_tsr import CUREDataset
from data.gtsrb import GTSRBSubsets
from data.imagenet import DistValSampler, BatchTransformDataLoader

def get_munit_loaders(args):
    """Return dataloaders based on command line argument for dataset."""

    if args.dataset == 'svhn':
        return get_svhn_loaders(args)
    elif args.dataset == 'gtsrb':
        return get_gtsrb_loaders(args)
    elif args.dataset == 'cure_tsr':
        return get_cure_tsr_loaders(args)
    else:
        raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

def get_svhn_loaders(args):
    """Get loaders with SVHN data for training MUNIT."""

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

    train_A = CUREDataset('train', args.source_of_nat_var, level=0,
                            args=args, return_labels=False)
    train_B = CUREDataset('train', args.source_of_nat_var, level=5,
                            args=args, return_labels=False)
    test_A = CUREDataset('test', args.source_of_nat_var, level=0,
                            args=args, return_labels=False)
    train_B = CUREDataset('test', args.source_of_nat_var, level=5,
                            args=args, return_labels=False)

    return _to_loader(train_A)

def _to_loader(*args):
    """Turn datasets into dataloaders for MUNIT."""

    def make_loader(dataset):
        return DataLoader(dataset, batch_size=1, shuffle=True)

    return [make_loader(d) for d in args]
        