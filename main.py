"""
Main
"""
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random

from model import VAE
from model import L2E
from model import Discriminator
from model import vgg_model
from train import train_models
from opts import parse_opts
args = parse_opts()


''' Seed Settings '''
# ---------------------------------------
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = True
np.random.seed(args.seed)
# ---------------------------------------


def main():

    print('-----------------------------------------------------------------------------------')
    print('-----Hybrid Deep Generative Prediction Network in Metal Additive Manufacturing-----')
    print('-----------------------------------------------------------------------------------')

    ''' Device '''
    # -----------------------------------------------------------------------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # -----------------------------------------------------------------------------------------

    ''' HDGPN Models '''
    # -----------------------------------------------------------------------------------------
    vae = VAE.VAE(nc=args.image_ch, ch_e=args.ch_encoder, ch_d=args.ch_decoder, nz=args.nz)
    D = Discriminator.Discriminator(nc=args.image_ch, ch=args.ch_discriminator)
    l2e = L2E.L2E(nl=args.nl, nz=args.nz)

    vae = vae.cuda()
    l2e = l2e.cuda()
    D = D.cuda()

    vae = nn.DataParallel(vae)
    D = nn.DataParallel(D)
    l2e = nn.DataParallel(l2e)
    # -----------------------------------------------------------------------------------------

    ''' Pre-trained VGG '''
    # -----------------------------------------------------------------------------------------
    vgg = vgg_model.vgg
    pretrained_data = torch.load('model/vgg_normalized.pth')
    vgg.load_state_dict(pretrained_data)
    if args.image_ch == 1:
        vgg[0] = nn.Conv2d(1, 1, (1, 1))
        vgg[2] = nn.Conv2d(1, 64, (3, 3))

    # freeze all VGG params
    for param in vgg.parameters():
        param.requires_grad_(False)
    vgg = vgg.cuda()
    vgg = nn.DataParallel(vgg)
    vgg = vgg.module
    # -----------------------------------------------------------------------------------------

    ''' Train models '''
    # -----------------------------------------------------------------------------------------
    vae, l2e, D = train_models(device, vae, l2e, D, vgg)
    # -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()