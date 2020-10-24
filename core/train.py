import torch
from torchvision.utils import save_image
from apex import amp
from datetime import datetime
import time
import os

import utils.dist_utils as dist_utils
from utils.logger import TensorboardLogger, FileLogger
from utils.meter import AverageMeter, NetworkMeter, TimeMeter
from utils.arg_parser import get_parser
from utils.saver import Saver, save_eval_df
from models.load import load_model
from classifiers.load import init_classifier
from training.utils import distributed_predict, accuracy, correct
from training.scheduler import Scheduler
from data.dataloader import get_loaders
from training.train_algs import *

args = get_parser()
is_master, is_rank0 = dist_utils.whoami(args)
args.world_size = dist_utils.env_world_size()
args.rank = dist_utils.env_rank()

# Only want master rank logging to tensorboard
tb = TensorboardLogger(args.logdir, is_master=is_master)
log = FileLogger(args.logdir, is_master=is_master, is_rank0=is_rank0)

def main():

    tb.log('sizes/world', dist_utils.env_world_size())
    dist_utils.setup_dist_backend(args)

    # load datasets, initialize classifiers, load model of natural variation
    trn_loader, val_loader, trn_samp, val_samp = get_loaders(args)
    model, criterion, optimizer = init_classifier(args)
    G = load_model(args, reverse=False)

    # create directory to save outputs and save images
    os.makedirs(args.save_path, exist_ok=True)
    save_images(trn_loader, val_loader, G, args)
    if args.delta_dim == 2: save_grid(trn_loader, G)

    # global start time for training
    start_time = datetime.now()

    # reload classifier from checkpoint if --resume flag is given
    if args.resume: reload_from_cpkt(model, optimizer, args)

    # Evaluate classifier on validation set and quit
    if args.evaluate: 
        top1, top5 = validate(val_loader, model, criterion, 0, start_time)
        if args.local_rank == 0: 
            save_eval_df(top1, top5, args)
            print(f'Top1: {top1} | Top5: {top5}')
        return

    if args.distributed: dist_utils.sync_processes(args)

    scheduler = Scheduler(optimizer, args, tb, log)
    saver = Saver(args, scheduler.tot_epochs)

    # main training loop
    best_top1 = 0.
    for epoch in range(args.start_epoch, scheduler.tot_epochs):

        if args.distributed is True:
            trn_samp.set_epoch(epoch)
            val_samp.set_epoch(epoch)

        train(trn_loader, model, criterion, optimizer, scheduler, epoch, G, args)
        top1, top5 = validate(val_loader, model, criterion, epoch, start_time)
        saver.update(top1, top5)
        time_diff = (datetime.now()-start_time).total_seconds()/3600.0

        log.event("~~epoch\t\thours\t\ttop1\t\ttop5")
        log.event(f"~~{epoch}\t\t{time_diff:.5f}\t\t{top1:.3f}\t\t{top5:.3f}\n")

        if top1 > best_top1:
            if is_rank0 is True: save_checkpoint(epoch, model, optimizer, args)
            best_top1 = top1

def train(trn_loader, model, criterion, optimizer, scheduler, epoch, G, args):
    """Train the classifier for a single epoch.
    
    Params:
        trn_loader: Loader for training set.
        model: Classifier instance.
        criterion: Loss function.
        optimizer: Optimization algorithm to use for training.
        scheduler: Learning rate scheduler.
        epoch: Current training epoch.
        G: Model of natural variation.
        args: Command line arguments.
    """
    
    net_meter = NetworkMeter()
    timer = TimeMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    for i, (imgs, target) in enumerate(trn_loader):
        if args.short_epoch and (i > 10): break

        imgs, target = imgs.cuda(), target.cuda()

        batch_num = i+1
        timer.batch_start()
        scheduler.update_lr(epoch, i+1, len(trn_loader))

        if args.mda is True:
            imgs, target = mda_train(imgs, target, model, G, args)

        elif args.mrt is True:
            imgs, target = mrt_train(imgs, target, model, criterion, G, args)

        elif args.mat is True:
            imgs, target = mat_train(imgs, target, model, criterion, G, args)

        elif args.pgd is True:
            imgs, target = pgd_train(imgs, target, model, criterion)

        output = model(imgs)
        loss = criterion(output, target)
                
        optimizer.zero_grad()
        if args.half_prec:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
               scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        timer.batch_end()
        corr1, corr5 = correct(output.data, target, topk=(1, 5))
        reduced_loss, batch_total = loss.data.item(), imgs.size(0)

        if args.distributed is True:
            metrics = torch.tensor([batch_total, reduced_loss, corr1, corr5]).float().cuda()
            batch_total, reduced_loss, corr1, corr5 = dist_utils.sum_tensor(metrics).cpu().numpy()
            reduced_loss = reduced_loss/dist_utils.env_world_size()
        else:
            corr1, corr5 = corr1.item(), corr5.item()

        top1acc = corr1 * (100.0 / batch_total)
        top5acc = corr5 * (100.0 / batch_total)

        losses.update(reduced_loss, batch_total)
        top1.update(top1acc, batch_total)
        top5.update(top5acc, batch_total)

        if should_print(batch_num, trn_loader, args) is True:
            tb.log_memory()
            tb.log_trn_times(timer.batch_time.val, timer.data_time.val, imgs.size(0))
            tb.log_trn_loss(losses.val, top1.val, top5.val)

            recv_gbit, transmit_gbit = net_meter.update_bandwidth()
            tb.log("sizes/batch_total", batch_total)
            tb.log('net/recv_gbit', recv_gbit)
            tb.log('net/transmit_gbit', transmit_gbit)
            
            output = (f'Epoch: [{epoch}][{batch_num}/{len(trn_loader)}]\t'
                      f'Time {timer.batch_time.val:.3f} ({timer.batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      f'Data {timer.data_time.val:.3f} ({timer.data_time.avg:.3f})\t'
                      f'BW {recv_gbit:.3f} {transmit_gbit:.3f}')
            log.verbose(output)

        tb.update_step_count(batch_total)


def validate(val_loader, model, criterion, epoch, start_time):
    """Run the validation set through a trained classifier.
    
    Params:
        val_loader: Loader for validation set.
        model: Classifier instance.
        criterion: Loss function.
        epoch: Current training epoch.
        start_time: Time when training started.
    """

    timer = TimeMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    eval_start_time = time.time()


    for i, (imgs,target) in enumerate(val_loader):

        imgs, target = imgs.cuda(), target.cuda()

        # terminates epoch early
        if args.short_epoch and (i > 10): break

        batch_num = i + 1
        timer.batch_start()

        if args.distributed:
            top1acc, top5acc, loss, batch_total = distributed_predict(imgs, target, model, criterion)
        else:
            with torch.no_grad():
                output = model(imgs)
                loss = criterion(output, target).data
            batch_total = imgs.size(0)
            top1acc, top5acc = accuracy(output.data, target, topk=(1,5))

        # Eval batch done. Logging results
        timer.batch_end()
        losses.update(loss, batch_total)
        top1.update(top1acc, batch_total)
        top5.update(top5acc, batch_total)

        if should_print(batch_num, val_loader, args) is True:
            output = (f'Test:  [{epoch}][{batch_num}/{len(val_loader)}]\t'
                      f'Time {timer.batch_time.val:.3f} ({timer.batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
            log.verbose(output)

    tb.log_eval(top1.avg, top5.avg, time.time()-eval_start_time)
    tb.log('epoch', epoch)

    return top1.avg, top5.avg

def should_print(batch_num, loader, args):
    """Checks whether logger should print output at current batch."""
    
    if (batch_num % args.print_freq == 0) or (batch_num == len(loader)):
        if args.local_rank == 0:
            return True
    return False

def save_checkpoint(epoch, model, optimizer, args):
    """Save checkpoint of trained classifier."""

    state = {
        'epoch': epoch+1, 'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }
    fname = os.path.join(args.save_path, f'classifier-checkpoint.tar')
    torch.save(state, fname)

def reload_from_cpkt(model, optimizer, args):
    """Reload classifier and optimizer from checkpoint."""

    checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.local_rank))
    model.load_state_dict(checkpoint['state_dict'])
    args.start_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])

def save_images(trn_loader, val_loader, G, args):
    """Save training, validation, and model-based images."""

    if args.local_rank == 0:
        trn_imgs, _ = next(iter(trn_loader))
        train_path = os.path.join(args.save_path, 'train.png')
        save_image(trn_imgs, train_path)

        val_imgs, _ = next(iter(val_loader))
        val_path = os.path.join(args.save_path, 'val.png')
        save_image(val_imgs, val_path)

        delta = torch.randn(trn_imgs.size(0), args.delta_dim, 1, 1).cuda()
        with torch.no_grad():
            out = G(trn_imgs.cuda(), delta)
        mb_path = os.path.join(args.save_path, 'model_based.png')
        save_image(out, mb_path)

def save_grid(trn_loader, G, lower=-1., upper=1., num_pts=10):
    """Grid nuisance space for 2-dimensional Delta spaces."""

    imgs, _ = next(iter(trn_loader))
    img = imgs[0].unsqueeze(0).cuda()
    img_samples = None

    for y in list(torch.linspace(lower, upper, steps=num_pts)):
        row_images = []
        for x in list(torch.linspace(lower, upper, steps=num_pts)):
            grid_style = torch.tensor([x, y]).reshape(1, 2).cuda().float()
            x_A_to_B = G(img, grid_style)
            row_images.append(x_A_to_B)

        row_sample = torch.cat(row_images, dim=-1)

        if img_samples is None:
            img_samples = row_sample
        else:
            img_samples = torch.cat([row_sample, img_samples], dim=-2)

    fname_orig = os.path.join(args.save_path, 'original_grid.png')
    save_image(img, fname_orig)

    fname_grid = os.path.join(args.save_path, 'generated_grid.png')
    save_image(img_samples, fname_grid)

if __name__ == '__main__':
    main()
    tb.close()


