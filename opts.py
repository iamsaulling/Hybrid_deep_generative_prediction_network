"""
Argument settings
"""
import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--root_path', type=str, default='', help='root path')
    parser.add_argument('--data_path', type=str, default='data/raw/', help='data path')
    parser.add_argument('--output_path', type=str, default='train results/', help='training output path')
    parser.add_argument('--seed', type=int, default=2021, metavar='S', help='random seed (default: 2021)')

    ''' Dataset Settings '''
    parser.add_argument('--data_width', type=int, default=8, help='length of velocity list')
    parser.add_argument('--power', type=list, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument('--power_min', type=float, default=150)
    parser.add_argument('--power_max', type=float, default=400)
    parser.add_argument('--velocity', type=list, default=[0.0 / 1400., 100. / 1400., 200. / 1400., 400. / 1400.,
                                                          700. / 1400., 900. / 1400., 1200. / 1400., 1400. / 1400.])
    parser.add_argument('--velocity_min', type=float, default=100)
    parser.add_argument('--velocity_max', type=float, default=1500)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--image_ch', type=int, default=1)
    parser.add_argument('--randomCrop_num', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=1920)
    parser.add_argument('--num_workers', type=int, default=1)

    ''' Model Settings '''
    parser.add_argument('--nz', type=int, default=64, help='dimensionality of latent feature space')
    parser.add_argument('--nl', type=int, default=2, help='dimensionality of input labels')
    parser.add_argument('--ch_encoder', type=int, default=64)
    parser.add_argument('--ch_decoder', type=int, default=64)
    parser.add_argument('--ch_discriminator', type=int, default=64)

    ''' Training Settings '''
    parser.add_argument('--train_epoch', type=int, default=500)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=100, help='frequency of saving checkpoints')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_vae', type=float, default=2e-5)
    parser.add_argument('--lr_l2e', type=float, default=2e-5)
    parser.add_argument('--lr_d', type=float, default=2e-5)
    parser.add_argument('--D_num_steps', type=int, default=5, help='number of D updates that triggers one update of G')

    ''' Weight Settings '''
    parser.add_argument('--kld_weight', type=float, default=0.001, help='weight for KLD loss')
    parser.add_argument('--gp_weight', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--aut_weight', type=float, default=0.001, help='weight for authenticity loss')
    parser.add_argument('--cstc_weight', type=float, default=0.1, help='weight for consistency loss')
    parser.add_argument('--vgg_weights', type=dict, default={'pool1': 1e7,
                                                             'pool2': 1e7,
                                                             'pool3': 1e7,
                                                             'pool4': 1e7,
                                                             'conv5_4': 5e6,
                                                             }, help='weight for vgg layers')

    ''' Test Settings '''
    parser.add_argument('--normalized_vgg', type=bool, default=True)
    parser.add_argument('--batch_size_test', type=int, default=64)

    args = parser.parse_args()

    return args