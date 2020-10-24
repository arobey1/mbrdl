"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import tensorboardX

from models.munit.utils import prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
from models.munit.trainer import MUNIT_Trainer
from data.munit_loaders import get_munit_loaders

import argparse
import torch.backends.cudnn as cudnn
import torch
import os
import sys
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='core/models/munit/munit.yaml', 
                        help='Path to the MUNIT config file.')
parser.add_argument('--output_path', type=str, default='./munit', 
                        help="Path where images/checkpoints will be saved")
parser.add_argument('--dataset', type=str, default='svhn', choices=['svhn', 'cure-tsr', 'gtsrb', 'imagenet', 'mnist'], 
                        help='Dataset to use for training MUNIT')
parser.add_argument('--data-size', type=int, default=32, 
                        help='Image size for training MUNIT')
parser.add_argument("--resume", action="store_true",
                        help='Resumes training from last avaiable checkpoint')
parser.add_argument('--source-of-nat-var', type=str, 
                        help='Type of natural variation to train on')
parser.add_argument('--train-data-dir', metavar='DIR', required=True, 
                        help='Path to training dataset (only used for ImageNet).')
parser.add_argument('--val-data-dir', metavar='DIR', 
                        help='Path to validation dataset (only used for ImageNet).')
args = parser.parse_args()

# Added for compatibility with train.py
args.local_rank = 0
args.setup_verbose = True

cudnn.benchmark = True

# Load experiment setting
config = get_config(args.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = args.output_path

# Setup model and data loader
device = torch.device('cuda')
trainer = torch.nn.DataParallel(MUNIT_Trainer(config))
trainer.to(device)

train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_munit_loaders(args)
train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(args.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(args.output_path + "/logs", model_name))
output_directory = os.path.join(args.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(args.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.module.resume(checkpoint_directory, hyperparameters=config) if args.resume else 0
while True:
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.module.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.module.dis_update(images_a, images_b, config)
            trainer.module.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.module.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.module.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.module.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.module.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
