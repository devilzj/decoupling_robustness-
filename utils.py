"""utils.py"""

import argparse
import subprocess
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from dataset import CustomImageFolder
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def get_mnist():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/youtu-face-identify-public/jiezhang/data', train=True, download=True,
                         transform=transforms.Compose(
                             [
                                 transforms.ToTensor(),
                             ])),
        batch_size=256, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/youtu-face-identify-public/jiezhang/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=256, shuffle=False, num_workers=4)
    return train_loader, test_loader

def get_celeba():
    root = "/youtu-face-identify-public/jiezhang/data/ft_local/celebA/"
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(), ])
    train_kwargs = {'root': root, 'transform': transform}
    dset = CustomImageFolder
    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=256,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader

    return data_loader

def get_dataset(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/youtu-face-identify-public/jiezhang/data', train=True, download=True,
                         transform=transforms.Compose(
                             [
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ])),
        batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/youtu-face-identify-public/jiezhang/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def where(cond, x, y):
    """Do same operation as np.where

    code from:
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond * x) + ((1 - cond) * y)


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay ' + str(delay) + ' -loop 0 ' + image_str + ' ' + output_gif
    subprocess.call(str1, shell=True)

def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss
