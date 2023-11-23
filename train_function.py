import torch
import torch.nn as nn
import numpy as np
from opts import parse_opts
args = parse_opts()


def kld_loss_function(mean, log_var, tar_mean):
    kld = torch.mean(-0.5 * torch.sum(1 + log_var - (mean - tar_mean).pow(2) - log_var.exp(), dim=1), dim=0)
    return kld


def generate_pred_labels(len, device):
    column_power = np.random.uniform(0.0, 1.0, size=(len, 1))
    column_velocity = np.random.uniform(0.0, 1.0, size=(len, 1))
    power_label = np.ones((len, 1)) * column_power
    power_label = np.clip(power_label, 0.0, 1.0)
    velocity_label = np.ones((len, 1)) * column_velocity
    velocity_label = np.clip(velocity_label, 0.0, 1.0)

    pred_labels = np.concatenate((power_label, velocity_label), axis=1)
    pred_labels = torch.from_numpy(pred_labels).type(torch.float).cuda()

    return pred_labels


def generate_sp_labels(len, p, v):
    power_label = np.ones((len, 1)) * p
    power_label = np.clip(power_label, 0.0, 1.0)
    velocity_label = np.ones((len, 1)) * v
    velocity_label = np.clip(velocity_label, 0.0, 1.0)

    sp_labels = np.concatenate((power_label, velocity_label), axis=1)
    sp_labels = torch.from_numpy(sp_labels).type(torch.float).cuda()

    return sp_labels


def get_features(image, model, layers=None):
    if layers is None:
        layers = {'7':  'pool1',
                  '14': 'pool2',
                  '27': 'pool3',
                  '40': 'pool4',
                  '51': 'conv5_4',
                  }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    n, d, h, w = tensor.size()
    features = tensor.view(n, d, h * w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (d * h * w)
    return gram


def rec_loss_function(raw_images, recon_images, vgg):
    raw_features = get_features(raw_images, vgg)
    recon_features = get_features(recon_images, vgg)

    style_loss_set = []
    for layer in raw_features:
        t_feature = raw_features[layer]
        t_gram = gram_matrix(t_feature).detach()

        rec_feature = recon_features[layer]
        rec_gram = gram_matrix(rec_feature)

        layer_style_loss = args.vgg_weights[layer] * nn.MSELoss()(t_gram, rec_gram)
        style_loss_set.append(layer_style_loss)

    return style_loss_set

