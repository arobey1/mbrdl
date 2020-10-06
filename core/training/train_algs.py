import torch
import torch.nn.functional as F

def mda_train(images, target, model, G, args):
    """Model-based Data Augmentation training algorithm.
    
    Params:
        images: Batch of training imagse.
        target: Labels corresponding to training batch.
        model: Classifier instance.
        G: Model of natural variation.
        args: Command line arguments.

    Returns:
        Images and target augmented with model-based data.
    """

    all_mb_images = [images]

    for _ in range(args.k):
        with torch.no_grad():
            delta = torch.rand(images.size(0), args.delta_dim, 1, 1).cuda()
            mb_images = G(images, delta)
            all_mb_images.append(mb_images)

    images = torch.cat(all_mb_images, dim=0).cuda()
    target = torch.cat([target for _ in range(len(all_mb_images))])

    return images, target

def mrt_train(images, target, model, criterion, G, args):
    """Model-based Robust Training training algorithm.
    
    Params:
        images: Batch of training imagse.
        target: Labels corresponding to training batch.
        model: Classifier instance.
        criterion: Loss function.
        G: Model of natural variation.
        args: Command line arguments.

    Returns:
        Images and target augmented with model-based data.
    """

    max_loss, worst_imgs = torch.tensor(0.0).cuda(), None

    for _ in range(args.k):
        with torch.no_grad():
            delta = torch.rand(images.size(0), args.delta_dim, 1, 1).cuda()
            mb_images = G(images, delta)
            mb_output = model(mb_images)
            mb_loss = criterion(mb_output, target)
            if mb_loss > max_loss:
                worst_imgs = mb_images
                max_loss = mb_loss

    images = torch.cat([images, worst_imgs.cuda()], dim=0)
    target = torch.cat([target, target])

    return images, target

def mat_train(images, target, model, criterion, G, args, alpha=0.1):
    """Model-based Adversarial Training training algorithm.
    
    Params:
        images: Batch of training imagse.
        target: Labels corresponding to training batch.
        model: Classifier instance.
        criterion: Loss function.
        G: Model of natural variation.
        args: Command line arguments.
        alpha: Step size for adversarial training.

    Returns:
        Images and target augmented with model-based data.
    """

    adv_delta = torch.zeros(images.size(0), args.delta_dim, 1, 1).cuda()
    adv_delta.requires_grad_(True)
    for _ in range(args.k):
        mb_images = G(images, adv_delta)
        loss = F.nll_loss(model(mb_images), target)
        loss.backward()
        grad_delta = adv_delta.grad.detach() # / torch.norm(adv_delta.grad.detach())
        adv_delta.data = (adv_delta + alpha * grad_delta).clamp(-1, 1)
        adv_delta.grad.zero_()

    adv_delta = adv_delta.detach().requires_grad_(False)
    mb_images = G(images, adv_delta)
    images = torch.cat([images, mb_images.cuda()], dim=0)
    target = torch.cat([target, target])

    return images, target

def pgd_train(images, target, model, criterion, num_iter=10, alpha=0.01, epsilon=8/255.):
    """PGD Adversarial training algorithm.
    
    Params:
        images: Batch of training imagse.
        target: Labels corresponding to training batch.
        model: Classifier instance.
        criterion: Loss function.
        num_iter: Number of steps of gradient ascent.
        alpha: Step size for gradient ascent.
        epsilon: Maximum (l_infinity) adversarial perturbation size.
    
    Adversarially perturbed data.
    """

    delta = torch.zeros_like(images, requires_grad=True)
    for t in range(num_iter):
        loss = criterion(model(images + delta), target)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach()).clamp(-epsilon, epsilon)
        delta.grad.zero_()

    return images + delta.detach(), target