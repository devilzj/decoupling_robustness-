"""solver.py"""
import sys
import warnings

warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import visdom

from torchvision import utils as vutils
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif, get_dataset, get_celeba, get_mnist
from model import BetaVAE, BetaVAE_C, BetaVAE_M, Adv_BetaVAE_M
from dataset import return_data
import ipdb

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class Solver(object):
    def __init__(self, args):
        self.use_cuda = True
        self.max_iter = args.max_iter  # maximum training iteration
        self.global_iter = 0

        self.z_dim = args.z_dim  # dimension of the representation z
        self.beta = args.beta  # beta-VAE
        self.gamma = args.gamma  # gamma parameter for KL-term in understanding beta-VAE
        self.C_stop_iter = args.C_stop_iter  # when to stop increasing the capacity
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1  # Adam optimizer beta1
        self.beta2 = args.beta2  # Adam optimizer beta2
        self.nc = 3
        self.args = args
        if args.dataset == "celeba":
            self.net = BetaVAE_C(self.z_dim, self.nc).cuda()
            self.data_loader = get_celeba()
        elif args.dataset == "cifar10":
            self.net = BetaVAE(self.z_dim, self.nc).cuda()
            self.data_loader, test_loader = get_dataset(args)
        elif args.dataset == "mnist":
            # self.net = BetaVAE_M(self.z_dim, self.nc).cuda()
            self.net = Adv_BetaVAE_M(self.z_dim, self.nc).cuda()
            self.data_loader, test_loader = get_mnist()
        else:
            raise NotImplementedError('not support!')
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                betas=(self.beta1, self.beta2))
        self.robust_model = args.robust_model

    def train(self):
        self.net.train()
        self.robust_model.eval()
        for epoch in tqdm(range(1000)):
            loss = 0.0
            for (data) in self.data_loader:
                if self.args.dataset == "celeba":
                    data = data.cuda()
                elif self.args.dataset == "mnist":
                    idx = torch.where(data[1] == 0)
                    data = data[0][idx].cuda()
                    out, feat = self.robust_model(data, return_features=True)
                else:
                    idx = torch.where(data[1] == 0)
                    data = data[0][idx].cuda()
                    # out, feat = self.robust_model(data, return_features=True)
                    # x_recon, mu, logvar = self.net(feat)  # feat:[64, 512]
                x_recon, mu, logvar = self.net(feat)
                # ipdb.set_trace()
                recon_loss = reconstruction_loss(data, x_recon, 'gaussian')
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                beta_vae_loss = recon_loss + self.beta * total_kld
                loss += beta_vae_loss.item()
                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()
            print("Loss :{}".format(loss / len(self.data_loader.dataset)))
            if self.args.dataset == "celeba":
                print("saving imgs...")
                save_image(data.clone(), 'ori{}.png'.format(self.args.dataset), normalize=True, scale_each=True, nrow=10)

                save_image(x_recon.clone(), 'syn{}.png'.format(self.args.dataset), normalize=True, scale_each=True,
                           nrow=10)
                print("saved already")

            else:
                if epoch % 10 == 0 and epoch != 0:
                    print("saving imgs...")
                    save_image(data.clone(), 'ori{}.png'.format(self.args.dataset), normalize=True, scale_each=True,
                               nrow=10)

                    save_image(x_recon.clone(), 'syn{}.png'.format(self.args.dataset), normalize=True, scale_each=True,
                               nrow=10)
                    print("saved already")
