import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from apex import amp
from apex.parallel import DistributedDataParallel

from classifiers.resnet import resnet50
from classifiers.basic import BasicNet

def init_classifier(args):
    """Load a classifier, model and initialize a loss function.
    
    Returns:
        model: Classifier instance.
        criterion: Loss function.
        optimizer: Optimization instance for training.    
    """
    
    if args.architecture == 'basic':
        model = BasicNet(num_classes=args.num_classes).cuda()

    # elif args.architecture == 'resnet50':
    #     model = resnet50(bn0=args.init_bn0, num_classes=args.num_classes, pretrained=args.pretrained).cuda()
            
    elif 'inception' in args.architecture:
        model = models.__dict__[args.architecture](pretrained=args.pretrained, num_classes=args.num_classes, aux_logits=False).cuda()

    else:
        model = models.__dict__[args.architecture](pretrained=args.pretrained, num_classes=args.num_classes).cuda()

    if args.optimizer == 'sgd':
        # start with 0 lr. Scheduler will change this later
        optimizer = optim.SGD(model.parameters(), 0, momentum=args.momentum, weight_decay=args.weight_decay) 
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    else:
        raise NotImplementedError(f'Optimizer {args.optimizer} not implemented')

    if args.half_prec: 
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    if args.distributed: 
        model = DistributedDataParallel(model)

    criterion = nn.CrossEntropyLoss().cuda()

    return model, criterion, optimizer

