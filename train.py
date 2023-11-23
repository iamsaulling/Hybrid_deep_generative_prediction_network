import os
import torch
import torch.nn as nn
import numpy as np
import timeit
import time
import matplotlib.pyplot as plt

from data_loader import generate_data
from data_loader import generate_dataloader
from utils import print_parameters
from utils import print_network
from utils import print_log
from utils import plot_rec_results
from utils import plot_pred_results

from train_function import kld_loss_function
from train_function import generate_pred_labels
from train_function import rec_loss_function

from opts import parse_opts
args = parse_opts()

# ------------------------------------------------------------------------------------
''' Parameters '''
train_epoch = args.train_epoch
resume_epoch = args.resume_epoch
save_freq = args.save_freq
nz = args.nz
nl = args.nl
batch_size = args.batch_size
lr_vae = args.lr_vae
lr_l2e = args.lr_l2e
lr_d = args.lr_d
D_num_steps = args.D_num_steps

# weight settings
kld_weight = args.kld_weight
gp_weight = args.gp_weight
aut_weight = args.aut_weight
cstc_weight = args.cstc_weight
# ------------------------------------------------------------------------------------


def train_models(device, vae, l2e, D, vgg):
    print('\n Start training HDGPN >>>\n')

    # ------------------------------------------------------------------------------------
    ''' Paths '''
    output_path = args.output_path

    log_path = output_path + 'logs/'
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    path_to_ckpt = output_path + 'checkpoint/'
    if not os.path.isdir(path_to_ckpt):
        os.makedirs(path_to_ckpt)

    path_to_train_result = output_path + 'train_result/'
    if not os.path.isdir(path_to_train_result):
        os.makedirs(path_to_train_result)
    # ------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------
    ''' Data Loader '''
    train_index_list = [0, 1, 2, 3, 5, 7,
                        8, 9, 11, 12, 14, 15,
                        16, 18, 20, 21, 23,
                        25, 26, 27, 28, 30,
                        32, 33, 34, 36, 37, 38, 39,
                        40, 42, 43, 45, 46]  # 70% training
    test_index_list = [4, 6, 10, 13, 17, 19, 22, 24, 29, 31, 35, 41, 44, 47]  # rest of groups for testing

    processed_data_path = 'data/process/'
    if not os.path.isdir(processed_data_path):
        os.makedirs(processed_data_path)
        generate_data()

    train_loader = generate_dataloader(train_index_list, args.batch_size, shuffle_signal=True)
    test_loader = generate_dataloader(test_index_list, args.batch_size, shuffle_signal=True)
    # -----------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    ''' Log Settings '''
    ctime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    log_file_name = ctime + '_Train log' + '.txt'

    ''' Print Parameter Settings '''
    print_parameters(log_path)

    ''' Print Models '''
    print_network(vae, 'vae', log_path)
    print_network(l2e, 'l2e', log_path)
    print_network(D, 'D', log_path)
    print_network(vgg, 'vgg', log_path)

    ''' Training History '''
    train_hist = {'VAE_loss': [],
                  'G_loss': [],
                  'L2E_loss': [],
                  'D_loss': [],
                  'WD': []
                  }

    ''' Optimizer '''
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=lr_vae)
    l2e_optimizer = torch.optim.Adam(l2e.parameters(), lr=lr_l2e)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # resume training
    if path_to_ckpt is not None and resume_epoch > 0:
        save_file = path_to_ckpt + 'ckpt_in_train/ckpt_epoch_{}.pth'.format(resume_epoch)
        checkpoint = torch.load(save_file)
        # load checkpoint
        vae.module.load_state_dict(checkpoint['VAE_state_dict'])
        l2e.module.load_state_dict(checkpoint['L2E_state_dict'])
        D.module.load_state_dict(checkpoint['D_state_dict'])
        vae_optimizer.load_state_dict(checkpoint['VAE_optimizer_state_dict'])
        l2e_optimizer.load_state_dict(checkpoint['L2E_optimizer_state_dict'])
        D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        # load training history
        train_hist = np.load(log_path + 'train_hist_{}.npy'.format(resume_epoch), allow_pickle=True).item()
    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # training start
    start_tmp = timeit.default_timer()
    for epoch in range(resume_epoch, train_epoch):
        vae.train()
        l2e.train()
        D.train()
        vgg.eval()

        # train_loader = generate_dataloader(train_index_list, args.batch_size, shuffle_signal=True)
        # some lists to store epoch losses
        epoch_vae_loss = []
        epoch_g_loss = []
        epoch_l2e_loss = []
        epoch_d_loss = []
        epoch_wd = []
        # ------------------------------------------------------------------------------------
        # iteration of train_loader
        for iter, (train_images, train_labels) in enumerate(train_loader):
            train_images = train_images.type(torch.float).cuda()
            train_groups = train_labels[:, nl]
            train_labels = train_labels[:, :nl].type(torch.float).cuda()
            mini_batch = train_images.size()[0]

            ''' Train D '''
            # loss with real images
            out_real = D(train_images)
            d_real = torch.mean(out_real)

            # loss with fake images
            recon_images, _, _, _ = vae(train_images)
            recon_images = recon_images.detach()  # reconstructed images
            pred_labels = generate_pred_labels(mini_batch, device)
            pred_means = l2e(pred_labels).detach()
            noises = torch.randn((mini_batch, nz)).cuda()
            pred_zs = pred_means + noises
            pred_images = vae.module.decoder(pred_zs).detach()  # predicted images

            out_recon = D(recon_images)
            out_pred = D(pred_images)
            d_fake= (torch.mean(out_pred) + torch.mean(out_recon)) / 2

            # gradient penalty
            alpha = torch.rand((mini_batch, 1, 1, 1)).cuda()
            beta = torch.rand((mini_batch, 1, 1, 1)).cuda()
            recon_x_hat = alpha * train_images.data + (1 - alpha) * recon_images.data
            pred_x_hat = beta * train_images.data + (1 - beta) * pred_images.data
            x_hat = torch.cat((recon_x_hat, pred_x_hat), dim=0)
            x_hat.requires_grad = True
            pred_hat = D(x_hat)

            gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat,
                                            grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = gp_weight * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

            D_loss = - d_real + d_fake + gradient_penalty
            Wasserstein_D = d_real - d_fake

            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            if ((iter + 1) % D_num_steps) == 0:
                ''' Train L2E '''
                _, means, _, _ = vae(train_images)
                means = means.detach()

                # prediction
                reg_means = l2e(train_labels)
                l2e_loss = nn.SmoothL1Loss()(means, reg_means)

                l2e_optimizer.zero_grad()
                l2e_loss.backward()
                l2e_optimizer.step()

                ''' Train VAE '''
                for para in vae.module.encoder.parameters():
                    para.requires_grad = True

                recon_images, means, log_vars, zs = vae(train_images)

                # reconstruction loss
                style_loss_set = rec_loss_function(train_images, recon_images, vgg)
                recon_loss = sum(style_loss_set)
                sty_loss1 = style_loss_set[0]
                sty_loss2 = style_loss_set[1]
                sty_loss3 = style_loss_set[2]
                sty_loss4 = style_loss_set[3]
                sty_loss5 = style_loss_set[4]

                # kld loss
                reg_means = l2e(train_labels).detach()
                kld_loss = kld_loss_function(means, log_vars, reg_means) * kld_weight

                vae_loss = recon_loss + kld_loss
                vae_optimizer.zero_grad()
                vae_loss.backward(retain_graph=True)

                for para in vae.module.encoder.parameters():
                    para.requires_grad = False

                # authenticity loss
                pred_labels = generate_pred_labels(mini_batch, device)
                pred_means = l2e(pred_labels).detach()
                noises = torch.randn((mini_batch, nz)).cuda()
                pred_zs = pred_means + noises
                pred_images = vae.module.decoder(pred_zs)

                out_real = D(train_images)
                g_real = torch.mean(out_real)

                out_recon = D(recon_images)
                out_pred = D(pred_images)
                g_fake = (torch.mean(out_recon) + torch.mean(out_pred)) / 2
                aut_loss = (g_real - g_fake) * aut_weight

                # consistence loss
                est_means, est_log_vars = vae.module.encoder(pred_images)
                cstc_loss = nn.MSELoss()(est_means, pred_means) * cstc_weight

                # G loss
                G_loss = aut_loss + cstc_loss
                G_loss.backward()

                vae_optimizer.step()

                # append epoch loss every 5 iteration
                epoch_vae_loss.append(vae_loss.item())
                epoch_g_loss.append(G_loss.item() / aut_weight)
                epoch_l2e_loss.append(l2e_loss.item())
                epoch_d_loss.append(D_loss.item())
                epoch_wd.append(Wasserstein_D.item())

                log = "Epoch[{:>3d}][{:>3d}/{:>3d}] " \
                      "VAE:{:.3f}[rec:{:.3f}, kld:{:.3f}], " \
                      "STYLE:[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}], " \
                      "G:{:.3f}[adv:{:.3f}, cstc:{:.3f}], " \
                      "L2E:{:.3f}, " \
                      "D:{:.1f}[r:{:.1f}, f:{:.1f}, gp:{:.1f}], " \
                      "WD:{:.1f}. " \
                      "Time:{:.0f}".\
                    format(epoch + 1, iter + 1, train_loader.dataset.__len__() // batch_size,
                           vae_loss.item(), recon_loss.item(), kld_loss.item(),
                           sty_loss1.item(), sty_loss2.item(), sty_loss3.item(), sty_loss4.item(), sty_loss5.item(),
                           G_loss.item(), aut_loss.item(), cstc_loss.item(),
                           l2e_loss.item(),
                           D_loss.item(), d_real.item(), d_fake.item(), gradient_penalty.item(),
                           Wasserstein_D.item(),
                           timeit.default_timer() - start_tmp)

                print(log)
                print_log(log, log_path, log_file_name)

        train_hist['VAE_loss'].append(np.mean(np.array(epoch_vae_loss)))
        train_hist['G_loss'].append(np.mean(np.array(epoch_g_loss)))
        train_hist['L2E_loss'].append(np.mean(np.array(epoch_l2e_loss)))
        train_hist['D_loss'].append(np.mean(np.array(epoch_d_loss)))
        train_hist['WD'].append(np.mean(np.array(epoch_wd)))
        # ------------------------------------------------------------------------------------
        # plot training history
        show_train_hist(train_hist, output_path)

        # ------------------------------------------------------------------------------------
        # model evaluation
        vae.eval()
        l2e.eval()
        D.eval()
        with torch.no_grad():
            recon_images, means, log_vars, zs = vae(train_images)

            ''' Plot Reconstruction Results '''
            fig_name = 'Epoch ' + str(epoch + 1) + '_rec_train.png'
            plot_rec_results(train_images, train_labels, recon_images, path_to_train_result, fig_name)

            ''' Plot Predicted Samples'''
            pred_labels = torch.tensor([[1.0, 0.0 / 1400.], [1.0, 1. / 14.], [1.0, 2. / 14.], [1.0, 4. / 14.],
                                        [1.0, 7. / 14.], [1.0, 9. / 14.], [1.0, 12. / 14.], [1.0, 14. / 14.],

                                        [0.8, 0.0 / 1400.], [0.8, 1. / 14.], [0.8, 2. / 14.], [0.8, 4. / 14.],
                                        [0.8, 7. / 14.], [0.8, 9. / 14.], [0.8, 12. / 14.], [0.8, 14. / 14.],

                                        [0.6, 0.0 / 1400.], [0.6, 1. / 14.], [0.6, 2. / 14.], [0.6, 4. / 14.],
                                        [0.6, 7. / 14.], [0.6, 9. / 14.], [0.6, 12. / 14.], [0.6, 14. / 14.],

                                        [0.4, 0.0 / 1400.], [0.4, 1. / 14.], [0.4, 2. / 14.], [0.4, 4. / 14.],
                                        [0.4, 7. / 14.], [0.4, 9. / 14.], [0.4, 12. / 14.], [0.4, 14. / 14.],

                                        [0.2, 0.0 / 1400.], [0.2, 1. / 14.], [0.2, 2. / 14.], [0.2, 4. / 14.],
                                        [0.2, 7. / 14.], [0.2, 9. / 14.], [0.2, 12. / 14.], [0.2, 14. / 14.],

                                        [0.0, 0.0 / 1400.], [0.0, 1. / 14.], [0.0, 2. / 14.], [0.0, 4. / 14.],
                                        [0.0, 7. / 14.], [0.0, 9. / 14.], [0.0, 12. / 14.], [0.0, 14. / 14.]],
                                       dtype=torch.float).cuda()

            for iter, (test_images, test_labels) in enumerate(test_loader):
                test_images = test_images.type(torch.float).cuda()
                test_groups = test_labels[:, nl]
                test_labels = test_labels[:, :nl].type(torch.float).cuda()
                mini_batch = test_images.size()[0]

                test_recon_images, test_means, test_log_vars, test_zs = vae(test_images)

                pred_means = l2e(pred_labels)
                noises = torch.randn(size=pred_means.shape).cuda()
                pred_zs = pred_means + noises

                if iter == 0:
                    fig_name = 'Epoch ' + str(epoch + 1) + '_rec_test.png'
                    plot_rec_results(test_images, test_labels, test_recon_images, path_to_train_result, fig_name)

                    pred_images = vae.module.decoder(pred_zs)
                    fig_name = 'Epoch ' + str(epoch + 1) + '_pred.png'
                    plot_pred_results(pred_labels, pred_images, path_to_train_result, fig_name)
        # ------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------
        # save checkpoint
        if (epoch + 1) % save_freq == 0 or epoch + 1 == train_epoch:
            save_file = path_to_ckpt + 'ckpt_in_train/ckpt_epoch_{}.pth'.format(epoch + 1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'epoch': epoch,
                    'VAE_state_dict': vae.module.state_dict(),
                    'L2E_state_dict': l2e.module.state_dict(),
                    'D_state_dict': D.module.state_dict(),
                    'VAE_optimizer_state_dict': vae_optimizer.state_dict(),
                    'L2E_optimizer_state_dict': l2e_optimizer.state_dict(),
                    'D_optimizer_state_dict': D_optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)

            np.save(log_path + 'train_hist_{}.npy'.format(epoch + 1), train_hist)
        # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # end for epoch

    return vae, l2e, D


def show_train_hist(hist, output_path):

    save_root = output_path + 'Train_hist.png'

    x = range(len(hist['D_loss']))

    y1 = hist['VAE_loss']
    y2 = hist['G_loss']
    y3 = hist['L2E_loss']
    y4 = hist['D_loss']
    y5 = hist['WD']

    # plt.plot(x, y1, label='VAE_loss')
    # plt.plot(x, y2, label='G_loss', color='darkblue')
    # plt.plot(x, y3, label='L2E_loss')
    plt.plot(x, y4, label='Loss of D', color='firebrick')
    plt.plot(x, y5, label='EM distance', color='darkcyan')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc='upper right')
    plt.grid(True)
    # plt.title('')
    plt.tight_layout()
    plt.savefig(save_root)
    plt.close()