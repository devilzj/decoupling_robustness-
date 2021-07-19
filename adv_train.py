from __future__ import print_function
import argparse  # Python 命令行解析工具
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import numpy as np
import random
from tqdm import tqdm
from utils import trades_loss, get_dataset, get_mnist
from torch.autograd import Variable


class SmallCNN(nn.Module):
    def __init__(self, in_dim=1, n_class=10):
        super(SmallCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 200),
            nn.Linear(200, n_class))

    def forward(self, x, return_features=False):
        out = self.conv(x)  # [256, 16, 5, 5]
        feat = out.view(out.size(0), -1)
        out = self.fc(feat)
        if return_features:
            return out, feat
        else:
            return out



def adv_train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        # calculate robust loss
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--weight_decay', '--wd', default=2e-4,
                        type=float, metavar='W')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--epsilon', default=0.3,
                        help='perturbation')
    parser.add_argument('--num_steps', default=40,
                        help='perturb number of steps')
    parser.add_argument('--step_size', default=0.01,
                        help='perturb step size')
    parser.add_argument('--beta', default=1.0,
                        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model', default="Resnet34", type=str,
                        help='save frequency')
    parser.add_argument('--type', default="normal", type=str,
                        help='save frequency')
    parser.add_argument('--other', default="", type=str,
                        help='save frequency')

    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 55:
        lr = args.lr * 0.1
    if epoch >= 75:
        lr = args.lr * 0.01
    if epoch >= 90:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _pgd_whitebox(model, X, y, epsilon, num_steps, random, step_size, **kwargs):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def eval_adv_test_whitebox_full(model, cfgs, test_loader, attack_method):
    """
    evaluate model by white-box attack
    """
    print("Evaluating {} Attack".format(attack_method))
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    params = dict(epsilon=cfgs['test_epsilon'], num_steps=cfgs['test_num_steps'],
                  step_size=cfgs['test_step_size'], num_classes=cfgs['num_classes'])
    total = len(test_loader.dataset)
    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data, requires_grad=True), Variable(target)
        if attack_method == 'PGD':
            err_natural, err_robust = _pgd_whitebox(model, X, y, random=True, **params)

        robust_err_total += err_robust

        natural_err_total += err_natural

    print('CLEAN : {:.4f}'.format(1 - natural_err_total / total))
    print('Robust : {:.4f}'.format(1 - robust_err_total / total))

    info = 'Evaluating {} Attack,CLEAN : {:.4f},  Robust : {:.4f}'.format(attack_method, 1 - natural_err_total / total,
                                                                          1 - robust_err_total / total)
    return info


if __name__ == '__main__':
    args = get_args()
    train_loader, test_loader = get_mnist()
    model = SmallCNN().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    bst_acc = -1
    cfgs = dict(random=True, test_num_steps=40, test_step_size=0.01, test_epsilon=0.3,
                num_classes=10)
    for epoch in tqdm(range(args.epochs)):
        adjust_learning_rate(optimizer, epoch)
        adv_train(args, model, train_loader, optimizer, epoch)
        if epoch % 20 == 0:
            info = eval_adv_test_whitebox_full(model, cfgs, test_loader, "PGD")

        torch.save(model.state_dict(), 'mnist_robust.pkl')
"""
python3 adv_train.py 
"""
