import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from opts import parse_opts
args = parse_opts()
black_threshold = (100 / 255.0 - 0.5) / 0.5

'''Print training settings'''


def print_network(net, name, log_path):
    ctime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    f = open(log_path + ctime + '_Network_' + name + '.txt', 'w')
    tmp = sys.stdout
    sys.stdout = f
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

    sys.stdout = tmp
    f.close()


def print_parameters(log_path):
    ctime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    with open(log_path + ctime + '_Parameters' + '.txt', 'w') as pf:
        pf.write('----------Overall Settings----------\n')
        pf.write('data_path: {}\n'.format(args.data_path))
        pf.write('seed: {}\n'.format(args.seed))
        pf.write('\n')

        pf.write('----------Dataset Settings----------\n')
        pf.write('image_size: {}\n'.format(args.image_size))
        pf.write('image_ch: {}\n'.format(args.image_ch))
        pf.write('randomCrop_num: {}\n'.format(args.randomCrop_num))
        pf.write('crop_size: {}\n'.format(args.crop_size))
        pf.write('\n')

        pf.write('----------Model Settings----------\n')
        pf.write('nz: {}\n'.format(args.nz))
        pf.write('nl: {}\n'.format(args.nl))
        pf.write('ch_encoder: {}\n'.format(args.ch_encoder))
        pf.write('ch_decoder: {}\n'.format(args.ch_decoder))
        pf.write('ch_discriminator: {}\n'.format(args.ch_discriminator))
        pf.write('\n')

        pf.write('----------Training Settings----------\n')
        pf.write('train_epoch: {}\n'.format(args.train_epoch))
        pf.write('resume_epoch: {}\n'.format(args.resume_epoch))
        pf.write('save_freq: {}\n'.format(args.save_freq))
        pf.write('batch_size: {}\n'.format(args.batch_size))
        pf.write('lr_vae: {}\n'.format(args.lr_vae))
        pf.write('lr_l2e: {}\n'.format(args.lr_l2e))
        pf.write('lr_d: {}\n'.format(args.lr_d))
        pf.write('D_num_steps: {}\n'.format(args.D_num_steps))
        pf.write('\n')

        pf.write('----------Weight Settings----------\n')
        pf.write('kld_weight: {}\n'.format(args.kld_weight))
        pf.write("gp_weight: {}\n".format(args.gp_weight))
        pf.write("aut_weight: {}\n".format(args.aut_weight))
        pf.write("cstc_weight: {}\n".format(args.cstc_weight))
        pf.write("vgg_weights: {}\n".format(args.vgg_weights))
        pf.write('\n')


def print_log(log, log_path, log_file_name):
    with open(log_path + log_file_name, "a") as pf:
        pf.write(log + "\n")


def plot_pred_results(pred_labels, pred_images, output_path, fig_name):
    plt.figure()
    plt.figure(figsize=(16, 12))

    for p in range(48):
        rp = p + 1
        power = pred_labels[p][0].item()
        velocity = pred_labels[p][1].item()

        # show predicted images
        plt.subplot(6, 8, rp)
        pred_image = pred_images[p].view(args.image_size, args.image_size).cpu().data.numpy()
        black = np.sum(pred_image <= black_threshold)
        porosity_rate = black / (args.image_size * args.image_size)
        if porosity_rate == 0:
            pred_image[0][0] = -1
        plt.imshow(pred_image, cmap='gray')
        plt.title('{:.0f}, {:.0f}, {:.2f}%'.format(power * (args.power_max - args.power_min) + args.power_min,
                                                   velocity * (args.velocity_max - args.velocity_min) + args.velocity_min,
                                                   porosity_rate * 100))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path + fig_name)
    plt.clf()
    plt.close('all')


def plot_rec_results(raw_images, raw_labels, recon_images, output_path, fig_name):
    plt.figure()
    plt.figure(figsize=(15, 15))
    position_index = [1, 2, 3, 4, 5, 6, 7, 8,
                      17, 18, 19, 20, 21, 22, 23, 24,
                      33, 34, 35, 36, 37, 38, 39, 40,
                      49, 50, 51, 52, 53, 54, 55, 56]
    sq = 8
    h = sq * sq // 2
    for p in range(h):
        rp = position_index[p]
        power = raw_labels[p][0].item()
        velocity = raw_labels[p][1].item()

        # show real images
        plt.subplot(sq, sq, rp)
        raw_image = raw_images[p].view(args.image_size, args.image_size).cpu().data.numpy()
        black = np.sum(raw_image <= black_threshold)
        porosity_rate = black / (args.image_size * args.image_size)
        if porosity_rate == 0:
            raw_image[0][0] = -1
        plt.imshow(raw_image, cmap='gray')
        plt.title('{:.0f}, {:.0f}, {:.2f}%'.format(power * (args.power_max - args.power_min) + args.power_min,
                                                 velocity * (args.velocity_max - args.velocity_min) + args.velocity_min,
                                                 porosity_rate * 100))
        plt.xticks([])
        plt.yticks([])

        # show reconstructed images
        plt.subplot(sq, sq, rp + sq)
        recon_image = recon_images[p].view(args.image_size, args.image_size).cpu().data.numpy()
        black = np.sum(recon_image <= black_threshold)
        porosity_rate = black / (args.image_size * args.image_size)
        if porosity_rate == 0:
            recon_image[0][0] = -1
        plt.imshow(recon_image, cmap='gray')
        plt.title('{:.2f}%'.format(porosity_rate * 100))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path + fig_name)
    plt.clf()
    plt.close('all')