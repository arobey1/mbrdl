import argparse

def get_parser():
    """Parse command line arguments and return args namespace."""

    parser = argparse.ArgumentParser(description='PyTorch training')
    
    parser.add_argument('--phases', type=str,
                    help='Specify epoch order of data resize and learning rate schedule: [{"ep":0,"sz":128,"bs":64},{"ep":5,"lr":1e-2}]')
    

    # paths to various directories
    parser.add_argument('--train-data-dir', metavar='DIR', required=True, 
                            help='Path to training dataset.')
    parser.add_argument('--val-data-dir', metavar='DIR', 
                            help='Path to validation dataset.')
    parser.add_argument('--save-path', type=str, 
                            help='Path for saving outputs')
    parser.add_argument('--logdir', default='', type=str,
                            help='Directory for tensorboard logs')
    parser.add_argument('--model-paths', type=str, nargs='*',
                            help="Path for model of natural variation")
    parser.add_argument('--config', type=str, default='core/models/munit/munit.yaml', 
                            help='Path to the MUNIT config file.')
    
    # training algorithms
    parser.add_argument('--mrt', action='store_true', 
                            help='Run Model-based Robust Training (MRT-k) with k = args.k')
    parser.add_argument('--mda', action='store_true', 
                            help='Run Model-based Data Augmentation (MDA-k) with k = args.k')
    parser.add_argument('--mat', action='store_true', 
                            help='RUN Model-based Adversarial Training (MAT-k) with k = args.k')
    parser.add_argument('-k', default=1, type=int, 
                            help='Hyperparameter k for model-based training')
    parser.add_argument('--pgd', action='store_true', 
                            help='Run PGD algorithm with default alpha = 0.01, epsilon = 8/255, n_steps = 20')
    
    # optimization settings and training parameters
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adadelta'], 
                            help='Optimization algorithm to use')
    parser.add_argument('--half-prec', action='store_true', 
                            help='Run model in half-precision mode using apex')
    parser.add_argument('--apex-opt-level', default='O1', type=str, 
                            help='opt_level for Apex amp initialization')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', 
                            help='weight decay (default: 1e-4)')
    parser.add_argument('--init-bn0', action='store_true', 
                            help='Intialize running batch norm mean to 0')
    parser.add_argument('--no-bn-wd', action='store_true', 
                            help='Remove batch norm from weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                            help='Momentum for SGD')
    parser.add_argument('--data-size', type=int, default=224, 
                            help="Size of each image")
    parser.add_argument('--batch-size', type=int, default=256, 
                            help='Training/validation batch size')
    parser.add_argument('--delta-dim', type=int, default=2, 
                            help="dimension of nuisance latent space")

    # architecture
    parser.add_argument('--architecture', default='resnet50', type=str, 
                            help='Architecture for classifier')
    parser.add_argument('--pretrained', action='store_true', 
                            help='Use pretrained model (only available for torchvision.models)')
    parser.add_argument('--num-classes', default=1000, type=int, 
                            help='Number of classes in datset')
    
    # dataset
    parser.add_argument('--dataset', required=True, type=str, choices=['imagenet', 'svhn', 'gtsrb', 'cure-tsr', 'mnist', 'mnist_c'],
                            help='Dataset to use for training/testing classifier.')
    parser.add_argument('--source-of-nat-var', type=str, 
                            help='Source of natural variation')

    # distributed setup
    parser.add_argument('--distributed', action='store_true', 
                            help='Run distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                            help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                            help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                            help='Used for multi-process training. Can either be manually set ' +
                            'or automatically set by using \'python -m multiproc\'.')

    # other parameters
    parser.add_argument('--print-freq', '-p', default=5, type=int, metavar='N', 
                            help='log/print every this many steps (default: 5)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
    parser.add_argument('--short-epoch', action='store_true', 
                            help='make epochs short (for debugging)')
    parser.add_argument('--setup-verbose', action='store_true', 
                            help='Print setup messages to console')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                            help='number of data loading workers (default: 8)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')

    args = parser.parse_args()

    return args